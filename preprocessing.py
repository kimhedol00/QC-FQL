import pickle
import numpy as np
import h5py
import os
import time
import glob
from multiprocessing import Pool, cpu_count
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
OUTPUT_DIR = './preprocessed_datasets_state'


# === MODIFIED: state 마스킹 유틸 ===
# state 구조 인덱스 매핑 (길이 38 가정: 0..37)
# 0:  left/gripper_pose (1)
# 1-3: left/tcp_force   (3)
# 4-9: left/tcp_pose    (6)
# 10-12: left/tcp_torque  (제거)
# 13-18: left/tcp_vel     (제거)
# 19:  right/gripper_pose (제거)
# 20-22: right/tcp_force  (제거)
# 23-28: right/tcp_pose   (제거)
# 29-31: right/tcp_torque (제거)
# 32-37: right/tcp_vel    (제거)
KEEP_IDXS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)

def mask_state_keep_left_pose_force_pose(state_vec: np.ndarray) -> np.ndarray:
    """
    입력 state_vec(1D)을 동일 길이로 유지하면서,
    left/gripper_pose(0), left/tcp_force(1..3), left/tcp_pose(4..9)만 보존.
    나머지 인덱스는 0으로 설정.
    """
    v = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    out = np.zeros_like(v, dtype=np.float32)
    # v 길이가 더 짧아도 안전하게 동작
    valid_keep = KEEP_IDXS[KEEP_IDXS < v.size]
    out[valid_keep] = v[valid_keep]
    return out
# === END MODIFIED ===


def convert_single_file(input_path: str) -> str:
    """
    [워커 함수] 단일 .pkl 파일을 완전한 .h5 파일로 변환하는 모든 작업을 수행합니다.
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

                # === MODIFIED: state 마스킹 적용 ===
                state_vec = obs['state'].flatten()
                state_vec = mask_state_keep_left_pose_force_pose(state_vec)
                obs_states.append(state_vec)
                # === END MODIFIED ===
                
                next_obs = step_data['next_observations']
                next_obs_head.append(np.squeeze(next_obs['left/head_cam'], axis=0))
                next_obs_wrist_l.append(np.squeeze(next_obs['left/wrist_cam'], axis=0))
                next_obs_wrist_r.append(np.squeeze(next_obs['right/wrist_cam'], axis=0))

                # === MODIFIED: next_state 마스킹 적용 ===
                next_state_vec = next_obs['state'].flatten()
                next_state_vec = mask_state_keep_left_pose_force_pose(next_state_vec)
                next_obs_states.append(next_state_vec)
                # === END MODIFIED ===
                
                actions.append(step_data['actions'].flatten())
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

        # 6. 최종 HDF5 파일로 저장
        with h5py.File(output_path, 'w') as f:
            for key, data in final_dataset.items():
                f.create_dataset(key, data=data, compression='gzip')
        
        return f"Success: {os.path.basename(input_path)} -> {os.path.basename(output_path)}"

    except Exception as e:
        return f"Failed: {os.path.basename(input_path)} | Error: {e}\n{traceback.format_exc()}"

def main():
    start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_paths = []
    print("Searching for .pkl files...")
    for input_dir in INPUT_DIRS:
        print(f"  - Searching in '{input_dir}'...")
        found_files = glob.glob(os.path.join(input_dir, '**', '*.pkl'), recursive=True)
        all_paths.extend(found_files)
        
    if not all_paths:
        print("Error: No .pkl files found in any of the specified directories.")
        return
    
    print(f"Found a total of {len(all_paths)} files. Starting serial conversion (one file at a time)...")
    
    results = []
    for path in tqdm(all_paths, desc="Converting Files"):
        result = convert_single_file(path)
        results.append(result)
    
    print("\n--- Conversion Summary ---")
    for res in sorted(results):
        print(res)
    
    end_time = time.time()
    print(f"\n✅ All files processed! Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
