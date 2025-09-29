import pickle
import numpy as np
import h5py
import os
import time
import glob
import argparse
from tqdm import tqdm
import traceback

# --- 설정 (사용자 수정 필요) ---

# 1. 원본 .pkl 파일들이 포함된 폴더 경로 목록
INPUT_DIRS = [
    '/home/work/AGI/QC-FQL/data/cleanup_table_two_waste/2025-08-19',
    '/home/work/AGI/QC-FQL/data/cleanup_table/2025-08-19',
    '/home/work/AGI/QC-FQL/data/cleanup_random/2025-08-20',
]

# 2. 전처리된 .h5 파일들이 저장될 폴더 경로
OUTPUT_DIR = './preprocessed_datasets_state_masked'

# ---- 기본값(기존 동작 유지용) ----
DEFAULT_STATE_KEEP = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)  # 기존 코드와 동일
# ACTION(길이 14): 기본적으로 마스킹 안 함
DEFAULT_ACTION_KEEP = np.array([0, 1, 2, 6])  # None이면 원본 유지


# ========== 공통 유틸 ==========
def parse_index_spec(spec: str):
    """
    '0,2,5-9' 같은 문자열을 정수 인덱스 리스트로 파싱.
    빈 문자열이나 None -> None 반환(마스킹 지정 없음).
    """
    if spec is None:
        return None
    spec = spec.strip()
    if spec == "":
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(b, a + 1))
        else:
            out.append(int(part))
    # 중복 제거 & 정렬
    return sorted(set(out))


def mask_vector_by_keep_or_drop(vec: np.ndarray, keep_idxs=None, drop_idxs=None) -> np.ndarray:
    """
    vec(1D)을 동일 길이 유지.
    - keep_idxs가 주어지면: 해당 인덱스만 보존, 나머지 0
    - keep_idxs가 None이고 drop_idxs가 주어지면: drop_idxs만 0, 나머지 보존
    - 둘 다 None: 원본 반환(복사본)
    out-of-range 인덱스는 자동 무시.
    """
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = v.size
    out = np.array(v, copy=True)

    if keep_idxs is not None:
        out[:] = 0.0
        valid_keep = [i for i in keep_idxs if 0 <= i < n]
        if valid_keep:
            out[valid_keep] = v[valid_keep]
        return out

    if drop_idxs is not None:
        valid_drop = [i for i in drop_idxs if 0 <= i < n]
        if valid_drop:
            out[valid_drop] = 0.0
        return out

    return out  # no mask


def arr1d(x):
    """(1, D) 형태도 flatten해서 1D로."""
    return np.asarray(x, dtype=np.float32).reshape(-1)


# ========== 변환 ==========
def convert_single_file(input_path: str,
                        state_keep=None,
                        state_drop=None,
                        action_keep=None,
                        action_drop=None) -> str:
    """
    단일 .pkl -> .h5 변환(+ state/action 마스킹)
    """
    try:
        # 1. 출력 파일 경로 생성
        base_name = os.path.basename(input_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{file_name_no_ext}.h5")

        # 2. 원본 데이터 로드
        with open(input_path, 'rb') as f:
            transitions = pickle.load(f)

        # 3. 데이터를 에피소드로 분할
        episodes = []
        current_episode = []
        for transition in transitions:
            current_episode.append(transition)
            if transition.get("infos", {}).get("succeed", False):
                episodes.append(current_episode)
                current_episode = []
        if current_episode:
            episodes.append(current_episode)

        if not episodes:
            return f"Skipped (no episodes): {input_path}"

        # 4. 각 에피소드 데이터 추출 및 변환
        processed_episodes = []
        for episode in episodes:
            obs_head, obs_wrist_l, obs_wrist_r, obs_states = [], [], [], []
            next_obs_head, next_obs_wrist_l, next_obs_wrist_r, next_obs_states = [], [], [], []
            actions, rewards, dones = [], [], []

            for step_data in episode:
                obs = step_data['observations']
                obs_head.append(np.squeeze(obs['left/head_cam'], axis=0))
                obs_wrist_l.append(np.squeeze(obs['left/wrist_cam'], axis=0))
                obs_wrist_r.append(np.squeeze(obs['right/wrist_cam'], axis=0))

                # --- state 마스킹 ---
                state_vec = arr1d(obs['state'])
                state_vec = mask_vector_by_keep_or_drop(
                    state_vec, keep_idxs=state_keep, drop_idxs=state_drop
                )
                obs_states.append(state_vec)

                next_obs = step_data['next_observations']
                next_obs_head.append(np.squeeze(next_obs['left/head_cam'], axis=0))
                next_obs_wrist_l.append(np.squeeze(next_obs['left/wrist_cam'], axis=0))
                next_obs_wrist_r.append(np.squeeze(next_obs['right/wrist_cam'], axis=0))

                # --- next_state 마스킹 ---
                next_state_vec = arr1d(next_obs['state'])
                next_state_vec = mask_vector_by_keep_or_drop(
                    next_state_vec, keep_idxs=state_keep, drop_idxs=state_drop
                )
                next_obs_states.append(next_state_vec)

                # --- action 마스킹 ---
                act_vec = arr1d(step_data['actions'])
                act_vec = mask_vector_by_keep_or_drop(
                    act_vec, keep_idxs=action_keep, drop_idxs=action_drop
                )
                actions.append(act_vec)

                rewards.append(step_data['rewards'])
                dones.append(step_data['dones'])

            processed_episodes.append({
                'observations/image_head': np.array(obs_head, dtype=np.uint8),
                'observations/image_wrist_left': np.array(obs_wrist_l, dtype=np.uint8),
                'observations/image_wrist_right': np.array(obs_wrist_r, dtype=np.uint8),
                'observations/state': np.array(obs_states, dtype=np.float32),
                'actions': np.array(actions, dtype=np.float32),
                'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1),
                'next_observations/image_head': np.array(next_obs_head, dtype=np.uint8),
                'next_observations/image_wrist_left': np.array(next_obs_wrist_l, dtype=np.uint8),
                'next_observations/image_wrist_right': np.array(next_obs_wrist_r, dtype=np.uint8),
                'next_observations/state': np.array(next_obs_states, dtype=np.float32),
                'terminals': np.array(dones, dtype=np.bool_).reshape(-1, 1),
            })

        # 5. 한 파일 내의 모든 에피소드 데이터를 하나로 합치기
        final_dataset = {}
        keys = processed_episodes[0].keys()
        for key in keys:
            arrays_to_concat = [ep[key] for ep in processed_episodes]
            final_dataset[key] = np.concatenate(arrays_to_concat, axis=0)

        # 6. 최종 HDF5 파일로 저장 (+마스킹 설정 메타데이터 기록)
        with h5py.File(output_path, 'w') as f:
            for key, data in final_dataset.items():
                f.create_dataset(key, data=data, compression='gzip')

            # 마스킹 설정을 파일 attribute로 기록(재현성)
            f.attrs['state_keep'] = np.array(state_keep if state_keep is not None else [], dtype=np.int32)
            f.attrs['state_drop'] = np.array(state_drop if state_drop is not None else [], dtype=np.int32)
            f.attrs['action_keep'] = np.array(action_keep if action_keep is not None else [], dtype=np.int32)
            f.attrs['action_drop'] = np.array(action_drop if action_drop is not None else [], dtype=np.int32)

        return f"Success: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"

    except Exception as e:
        return f"Failed: {os.path.basename(input_path)} | Error: {e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(description="Convert raw .pkl to preprocessed .h5 with optional masking")
    ap.add_argument("--state-keep", type=str, default=None,
                    help="state에서 보존할 인덱스(e.g., '0-6,10,12'). 지정 시 drop 무시. 기본: '0-6'")
    ap.add_argument("--state-drop", type=str, default=None,
                    help="state에서 제거할 인덱스(e.g., '10-37'). keep이 없을 때만 적용")
    ap.add_argument("--action-keep", type=str, default=None,
                    help="action에서 보존할 인덱스(e.g., '0-6,7-13'). 지정 시 drop 무시")
    ap.add_argument("--action-drop", type=str, default=None,
                    help="action에서 제거할 인덱스(e.g., '3-5,10-12')")
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="출력 폴더")
    ap.add_argument("--limit-files", type=int, default=0, help="최대 변환 파일 수 (0=제한 없음)")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 인덱스 파싱
    state_keep = parse_index_spec(args.state_keep)
    state_drop = parse_index_spec(args.state_drop)
    action_keep = parse_index_spec(args.action_keep)
    action_drop = parse_index_spec(args.action_drop)

    # 기본값(기존 동작) 적용 로직
    if state_keep is None and state_drop is None:
        state_keep = list(DEFAULT_STATE_KEEP)  # 기존: 0..6만 보존

    if action_keep is None and action_drop is None and DEFAULT_ACTION_KEEP is not None:
        action_keep = list(DEFAULT_ACTION_KEEP)

    # 파일 수집
    all_paths = []
    print("Searching for .pkl files...")
    for input_dir in INPUT_DIRS:
        print(f"  - Searching in '{input_dir}'...")
        found_files = glob.glob(os.path.join(input_dir, '**', '*.pkl'), recursive=True)
        all_paths.extend(found_files)

    if not all_paths:
        print("Error: No .pkl files found in any of the specified directories.")
        return

    if args.limit_files > 0:
        all_paths = all_paths[:args.limit_files]

    # 설정 에코
    print("\n[Mask Config]")
    print(f"  state_keep:  {state_keep}")
    print(f"  state_drop:  {state_drop}")
    print(f"  action_keep: {action_keep}")
    print(f"  action_drop: {action_drop}")
    print(f"\nFound a total of {len(all_paths)} files. Starting serial conversion...\n")

    start_time = time.time()
    results = []
    for path in tqdm(all_paths, desc="Converting Files"):
        res = convert_single_file(path,
                                  state_keep=state_keep,
                                  state_drop=state_drop,
                                  action_keep=action_keep,
                                  action_drop=action_drop)
        results.append(res)

    print("\n--- Conversion Summary ---")
    for res in sorted(results):
        print(res)

    elapsed = time.time() - start_time
    print(f"\n✅ All files processed! Total time taken: {elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
