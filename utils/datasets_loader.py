
import glob, tqdm, os

import numpy as np

import h5py

# 워커 프로세스 초기화 함수
def init_worker(data_dict, keys_list):
    """
    각 워커 프로세스가 생성될 때 한 번만 호출되어,
    전역 변수로 큰 데이터 객체를 설정합니다.
    """
    global final_data_global, all_dataset_keys_global
    final_data_global = data_dict
    all_dataset_keys_global = keys_list

# 병렬 작업을 수행할 워커 함수
def load_chunk(task):
    """
    하나의 HDF5 파일을 읽어, initializer를 통해 받은 전역 배열에 데이터를 채워 넣습니다.
    """
    path, start_idx, size = task
    end_idx = start_idx + size
    try:
        with h5py.File(path, 'r') as f:
            for key in all_dataset_keys_global:
                final_data_global[key][start_idx:end_idx] = f[key][()]
        return size  # 작업한 크기 반환
    except Exception as e:
        print(f"\n[경고] 파일 처리 중 오류 '{path}': {e}")
        return 0 # 오류 발생 시 0 반환

def load_dataset_dir(path, num_workers = 1):
    # 1. 모든 데이터셋의 키와 최종 크기를 미리 계산합니다.
    print("Step 1/3: Calculating total dataset size...")
    dataset_paths = sorted(glob.glob(os.path.join(path, '*.h*5')))
    if not dataset_paths:
        raise FileNotFoundError(f"No HDF5 files found in {path}")
    
    # 유효한 파일 목록과 전체 크기를 저장할 변수
    valid_dataset_paths = []
    total_size = 0
    all_dataset_keys = []
    
    # 첫 번째 유효한 파일을 찾아 데이터 구조 파악
    example_file = None
    for path in dataset_paths:
        try:
            example_file = h5py.File(path, 'r')
            def find_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    all_dataset_keys.append(name)
            example_file.visititems(find_keys)
            break # 성공하면 루프 탈출
        except OSError:
            print(f"\n[경고] 파일 '{path}'이(가) 손상되어 건너뜁니다.")
    
    if example_file is None:
        raise ValueError("모든 데이터셋 파일이 손상되었거나 읽을 수 없습니다.")

    shapes = {key: example_file[key].shape[1:] for key in all_dataset_keys}
    dtypes = {key: example_file[key].dtype for key in all_dataset_keys}
    example_file.close()

    # 모든 파일 스캔하여 크기 합산
    for path in tqdm.tqdm(dataset_paths, desc="Scanning file sizes"):
        try:
            with h5py.File(path, 'r') as f:
                total_size += f['actions'].shape[0]
            valid_dataset_paths.append(path)
        except OSError as e:
            print(f"\n[경고] 파일 '{path}'을(를) 읽는 중 오류 발생: {e}. 건너뜁니다.")
    
    dataset_paths = valid_dataset_paths
    if not dataset_paths:
        raise FileNotFoundError("오류: 유효한 HDF5 데이터셋 파일을 하나도 찾을 수 없습니다.")

    print(f"Total timesteps found: {total_size}")

    # 2. 최종 데이터를 담을 비어있는 NumPy 배열을 미리 할당합니다.
    print("Step 2/3: Pre-allocating memory for the final dataset...")
    final_data = {
        key: np.empty((total_size, *shapes[key]), dtype=dtypes[key])
        for key in all_dataset_keys
    }

    workers = int(max(0, num_workers))
    mode_str = "sequential" if workers <= 1 else f"parallel ({workers} workers)"
    print(f"Step 3/3: Loading data in {mode_str}...")

    tasks = []
    current_idx = 0
    for path in dataset_paths:
        with h5py.File(path, 'r') as f:
            size = f['actions'].shape[0]
            tasks.append((path, current_idx, size))
            current_idx += size

    if workers <= 1:
        # === 단일 프로세스 순차 로딩 ===
        from tqdm import tqdm as _tqdm
        with _tqdm(total=total_size, desc="Sequential Loading") as pbar:
            for (path, start_idx, size) in tasks:
                end_idx = start_idx + size
                try:
                    with h5py.File(path, 'r') as f:
                        for key in all_dataset_keys:
                            final_data[key][start_idx:end_idx] = f[key][()]
                    pbar.update(size)
                except Exception as e:
                    print(f"\n[경고] 파일 처리 중 오류 '{path}': {e}")
    else:
        # === 멀티프로세싱 로딩 ===
        from multiprocessing import Pool
        with tqdm.tqdm(total=total_size, desc="Parallel Loading") as pbar:
            # 필요한 만큼만 프로세스 생성
            with Pool(processes=workers, initializer=init_worker, initargs=(final_data, all_dataset_keys)) as pool:
                for loaded_size in pool.imap_unordered(load_chunk, tasks):
                    pbar.update(loaded_size)

    # 5. 최종 train_dataset 딕셔너리를 재구성합니다.
    train_dataset = {
        'observations': {
            'image_head': final_data['observations/image_head'],
            'image_wrist_left': final_data['observations/image_wrist_left'],
            'image_wrist_right': final_data['observations/image_wrist_right'],
            'state': final_data['observations/state'],
        },
        'actions': final_data['actions'],
        'rewards': final_data['rewards'],
        'terminals': final_data['terminals'],
        'next_observations': {
            'image_head': final_data['next_observations/image_head'],
            'image_wrist_left': final_data['next_observations/image_wrist_left'],
            'image_wrist_right': final_data['next_observations/image_wrist_right'],
            'state': final_data['next_observations/state'],
        },
    }
    print("Dataset successfully merged.")
    # D4RL 데이터셋 형식에 맞게 'timeouts'와 'masks' 추가
    train_dataset['timeouts'] = train_dataset['terminals']
    train_dataset['masks'] = 1.0 - train_dataset['terminals']
    return train_dataset


def load_dataset_path(path):
    print(f"Loading custom dataset from: {path}")
        
    # 1. HDF5 파일에서 데이터 로드
    with h5py.File(path, 'r') as f:
        train_dataset = {
            'observations': {
                'image_head': f['observations/image_head'][()],
                'image_wrist': f['observations/image_wrist'][()],
                'state': f['observations/state'][()],
            },
            'actions': f['actions'][()],
            'rewards': f['rewards'][()],
            'terminals': f['terminals'][()],
            'next_observations': {
                'image_head': f['next_observations/image_head'][()],
                'image_wrist': f['next_observations/image_wrist'][()],
                'state': f['next_observations/state'][()],
            },
        }
    
    # D4RL 데이터셋 형식에 맞게 'timeouts'와 'masks' 추가
    train_dataset['timeouts'] = train_dataset['terminals']
    train_dataset['masks'] = 1.0 - train_dataset['terminals']
    return train_dataset
