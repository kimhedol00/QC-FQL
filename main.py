import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

import h5py

# --- MODIFICATION START: 워커 함수를 스크립트 최상위로 이동 ---

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

# --- MODIFICATION END ---



if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 200000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')


flags.DEFINE_string('custom_dataset_path', None, 'Path to a custom HDF5 dataset file.')
flags.DEFINE_string('custom_dataset_dir', None, 'Path to a directory containing custom HDF5 dataset parts.')

flags.DEFINE_integer('horizon_length', 10, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    
    # =================================================================================
    # Data Loading Section (OPTIMIZED FOR SPEED AND MEMORY)
    # =================================================================================
    if FLAGS.custom_dataset_dir is not None:
        # 1. 모든 데이터셋의 키와 최종 크기를 미리 계산합니다.
        print("Step 1/3: Calculating total dataset size...")
        dataset_paths = sorted(glob.glob(os.path.join(FLAGS.custom_dataset_dir, '*.h*5')))
        if not dataset_paths:
            raise FileNotFoundError(f"No HDF5 files found in {FLAGS.custom_dataset_dir}")
        
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

        # 3. 여러 CPU 코어를 사용해 병렬로 데이터를 읽고 미리 할당된 배열에 채워 넣습니다.
        print(f"Step 3/3: Loading data in parallel using {os.cpu_count()} cores...")
        
        tasks = []
        current_idx = 0
        for path in dataset_paths:
            with h5py.File(path, 'r') as f:
                size = f['actions'].shape[0]
                tasks.append((path, current_idx, size))
                current_idx += size
        
        # --- MODIFICATION START: Pool 생성 방식 변경 ---
        with tqdm.tqdm(total=total_size, desc="Parallel Loading") as pbar:
            from multiprocessing import Pool, Manager
            # Manager를 사용해 공유 가능한 객체 생성
            # initializer와 initargs를 사용해 각 워커에 데이터 전달
            with Pool(initializer=init_worker, initargs=(final_data, all_dataset_keys)) as pool:
                for loaded_size in pool.imap_unordered(load_chunk, tasks):
                    pbar.update(loaded_size)
        # --- MODIFICATION END ---

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

        # 2. 평가(Evaluation)를 위한 환경 생성
        # <<< CHANGED: 평가 간격(eval_interval)에 따라 환경 생성을 건너뛰도록 수정
        if FLAGS.eval_interval > 0:
            # 평가를 진행할 경우에만 환경을 생성
            print(f"Creating evaluation environment for: {FLAGS.env_name}")
            env, eval_env, _, val_dataset = make_env_and_datasets(FLAGS.env_name)
        else:
            # 평가를 안 할 경우, 환경 변수들을 None으로 설정
            print("Evaluation is disabled (eval_interval=0). Skipping environment creation.")
            env, eval_env, val_dataset = None, None, None

    # =================================================================================
    # Data Loading Section (CHANGED)
    # =================================================================================
    elif FLAGS.custom_dataset_path is not None:
        print(f"Loading custom dataset from: {FLAGS.custom_dataset_path}")
        
        # 1. HDF5 파일에서 데이터 로드
        with h5py.File(FLAGS.custom_dataset_path, 'r') as f:
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

        # 2. 평가(Evaluation)를 위한 환경 생성
        # <<< CHANGED: 평가 간격(eval_interval)에 따라 환경 생성을 건너뛰도록 수정
        if FLAGS.eval_interval > 0:
            # 평가를 진행할 경우에만 환경을 생성
            print(f"Creating evaluation environment for: {FLAGS.env_name}")
            env, eval_env, _, val_dataset = make_env_and_datasets(FLAGS.env_name)
        else:
            # 평가를 안 할 경우, 환경 변수들을 None으로 설정
            print("Evaluation is disabled (eval_interval=0). Skipping environment creation.")
            env, eval_env, val_dataset = None, None, None
            
    # data loading
    elif FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
