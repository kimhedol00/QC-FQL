import glob, tqdm, wandb, os, json, random, time, jax, threading, pickle, re
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from utils.flax_utils import restore_full_checkpoint, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer
from utils.async_saver import AsyncSaver
import agentlace.inference as ali

from evaluation import evaluate
from agents import agents
import numpy as np

import h5py

PROCESS_KEYS = {
    "state": "state",
    # For Single
    "head_cam": "image_head",
    # "front_cam": "image_front",
    "front_cam": "image_wrist_left",
    "wrist_cam": "image_wrist_right",
    # For Dual
    "right/head_cam": "image_head",
    # "right/front_cam": "image_front",
    "right/front_cam": "image_wrist_left",
    "right/wrist_cam": "image_wrist_right",
    "left/wrist_cam": "image_wrist_left",
}

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000000, 'Evaluation interval.')
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
flags.DEFINE_string('custom_dataset_dir', None, 'Path to a custom HDF5 dataset file.')
flags.DEFINE_string('ckpt_path', None, 'Path to pretrained checkpoint.')

flags.DEFINE_integer('horizon_length', 10, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_bool('override_aborts_chunk', True, 'Flush action chunk when intervening.')
flags.DEFINE_integer('min_offline_intervention_len', -1, 'min length to push into demo buffer (default = horizon_length)')
flags.DEFINE_integer('param_pull_interval', 500, 'Actor pulls latest params every N env steps.') # N step 마다 가장 최신 모델 load 해와서 환경에서 작동하도록 #[TODO] 적당히 해당 값 변경

flags.DEFINE_integer('trainer_rpc_port', 45587, 'Remote env server port.')

flags.DEFINE_integer(
    'data_loader_workers',
    0,
    "Number of CPU workers for dataset loading. 0 or 1 → sequential load; ≥2 → multiprocessing."
)

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

def extract_step(p):
    m = re.search(r'(?:checkpoint|params)_(\d+)\.pkl$', os.path.basename(p))
    return int(m.group(1)) if m else -1

def resolve_ckpt_path(path):
    """파일이면 그대로, 디렉토리면 최신 step의 checkpoint를 선택.
       우선순위: checkpoint_*.pkl(풀) > params_*.pkl(에이전트 전용)
    """
    if path is None:
        return None
    if os.path.isdir(path):
        c_full = glob.glob(os.path.join(path, 'checkpoint_*.pkl'))
        c_agent = glob.glob(os.path.join(path, 'params_*.pkl'))
        cands = sorted(c_full, key=extract_step) if c_full else sorted(c_agent, key=extract_step)
        if not cands:
            raise FileNotFoundError(f'No checkpoint under {path}')
        return cands[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent

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

        workers = int(max(0, FLAGS.data_loader_workers))
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
    
    elif FLAGS.custom_dataset_path is not None:
        print(f"Loading custom dataset from: {FLAGS.custom_dataset_path}")
        
        with h5py.File(FLAGS.custom_dataset_path, 'r') as f:
            train_dataset = {
                'observations': {
                    'image_head': f['observations/image_head'][()],
                    'image_wrist_left': f['observations/image_wrist_left'][()],
                    'image_wrist_right': f['observations/image_wrist_right'][()],
                    'state': f['observations/state'][()],
                },
                'actions': f['actions'][()],
                'rewards': f['rewards'][()],
                'terminals': f['terminals'][()],
                'next_observations': {
                    'image_head': f['next_observations/image_head'][()],
                    'image_wrist_left': f['next_observations/image_wrist_left'][()],
                    'image_wrist_right': f['next_observations/image_wrist_right'][()],
                    'state': f['next_observations/state'][()],
                },
            }
        
        # D4RL 데이터셋 형식에 맞게 'timeouts'와 'masks' 추가
        train_dataset['timeouts'] = train_dataset['terminals']
        train_dataset['masks'] = 1.0 - train_dataset['terminals']
    else:
        # raise ValueError("custom_dataset_path를 반드시 지정해야 합니다.")
        train_dataset = {}
        dummy_images = np.zeros((100, 256, 256, 3))
        dummy_states = np.zeros((100, 19))
        dummy_actions = np.zeros((100, 7))
        dummy_rewards = np.zeros((100, 1))
        dummy_terminals = np.zeros((100, 1))
        dummy_timeouts = np.zeros((100, 1))
        dummy_masks = np.zeros((100, 1))

        train_dataset = {
            'observations': {
                "image_head": dummy_images,
                "image_wrist_left": dummy_images,
                "image_wrist_right": dummy_images,
                "state": dummy_states
            },
            'actions': dummy_actions,
            'rewards': dummy_rewards,
            'terminals': dummy_terminals,
            'next_observations': {
                "image_head": dummy_images,
                "image_wrist_left": dummy_images,
                "image_wrist_right": dummy_images,
                "state": dummy_states
            },
            'timeouts': dummy_timeouts,
            'masks': dummy_masks
        }

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    log_step = 0
    rpc_call_count = 0
    action_queue = []
    local_version = -1
    
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
        
        # if is_robomimic_env(FLAGS.env_name):
        #     penalty_rewards = ds["rewards"] - 1.0
        #     ds_dict = {k: v for k, v in ds.items()}
        #     ds_dict["rewards"] = penalty_rewards
        #     ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

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

    rl_buffer = ReplayBuffer.create(example_batch, size=FLAGS.buffer_size)
    demo_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )

    rl_buffer_lock = threading.Lock()
    demo_buffer_lock = threading.Lock()

    log_lock = threading.Lock()
    rpc_lock = threading.Lock()
    intervene_lock = threading.Lock()
    action_queue_lock = threading.Lock()

    prev_step_intervened = False  # 직전 스텝에 유저 개입 있었는지
    
    # 파라미터 공유
    params_ref = {'params': agent.network.params}
    version_id = {'val': 0}
    params_lock = threading.Lock()
    stop_event = threading.Event()

    if FLAGS.ckpt_path:
        try:
            ckpt_path = resolve_ckpt_path(FLAGS.ckpt_path)

            with open(ckpt_path, 'rb') as _f:
                obj = pickle.load(_f)

            if isinstance(obj, dict) and ( # 학습 ckpt 불러오기
                'rl_buffer' in obj or 'meta' in obj or 'rng' in obj
            ):
                agent, meta = restore_full_checkpoint(agent, rl_buffer, demo_buffer, ckpt_path)

                log_step = int(meta.get('log_step', log_step))
                rpc_call_count = int(meta.get('rpc_call_count', 0))
                online_rng = meta.get("online_rng", online_rng)
                if meta.get('version_id') is not None:
                    version_id['val'] = int(meta['version_id'])

                ckpt_kind = "full"
            else: # agent param만 불러오기
                agent = restore_agent_with_file(agent, ckpt_path)
                ckpt_kind = "agent"

            with params_lock:
                params_ref['params'] = agent.network.params

            print(f"[RESTORE] {ckpt_kind} checkpoint loaded: {ckpt_path}")
            print(f"[RESTORE] log_step={log_step}, rpc_call_count={locals().get('rpc_call_count', 'n/a')}, version_id={version_id['val']}")

        except Exception as e:
            print(f"[RESTORE][WARN] 복구 실패: {e}. 새 학습으로 진행합니다.")

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()

    # Learner Thread
    def learner_loop():
        nonlocal agent, log_step
        H = FLAGS.horizon_length
        gamma = FLAGS.discount

        while not stop_event.is_set():
            if log_step < FLAGS.start_training:
                time.sleep(0.001)
                continue

            B = config['batch_size'] * FLAGS.utd_ratio
            # B_demo = B // 2
            B_demo = int(B * demo_buffer.size / (demo_buffer.size + rl_buffer.size))
            B_rl   = B - B_demo

            with demo_buffer_lock:
                demo_sample = demo_buffer.sample_sequence(B_demo, sequence_length=H, discount=gamma)
            with rl_buffer_lock:
                # demo_sample = rl_buffer.sample_sequence(B_demo, sequence_length=H, discount=gamma)
                rl_sample = rl_buffer.sample_sequence(B_rl, sequence_length=H, discount=gamma)

            # batch = {k: np.concatenate([demo_sample[k], rl_sample[k]], axis=0) for k in demo_sample}
            assert jax.tree_util.tree_structure(demo_sample) == jax.tree_util.tree_structure(rl_sample), "demo_sample과 rl_sample의 구조가 다릅니다."

            batch = jax.tree_util.tree_map(
                lambda a, b: np.concatenate([a, b], axis=0),
                demo_sample, rl_sample
            )
            batch = jax.tree.map(
                lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]),
                batch
            )

            new_agent, update_info = agent.batch_update(batch)

            # 최신 파라미터 업데이트
            with params_lock:
                agent = new_agent
                # params_ref['params'] = agent.network.params
                version_id['val'] += 1
                update_version = version_id['val']
                # print(f"[LOG] Param updated: Version {update_version}")

            logger.log(update_info, "online_agent", step=log_step)

    learner_thread = threading.Thread(target=learner_loop, name="learner", daemon=True)
    learner_thread.start()

    # 비동기 저장용
    async_saver = AsyncSaver(
        FLAGS.save_dir, params_lock,
        rl_buffer_lock=rl_buffer_lock,
        demo_buffer_lock=demo_buffer_lock
    )

    # Actor server 부분
    server = ali.InferenceServer(port_num=FLAGS.trainer_rpc_port)

    def _process_obs_keys(ob_dict):
        return {PROCESS_KEYS[k]: v for k, v in ob_dict.items() if k in PROCESS_KEYS}

    def on_reset(payload):
        nonlocal action_queue, prev_step_intervened
        with action_queue_lock:
            action_queue.clear()
        with intervene_lock:
            prev_step_intervened = False
        return {"ok": True}
    
    def act(payload):
        nonlocal online_rng, agent, action_queue, prev_step_intervened

        ob = payload["ob"]
        processed_ob = _process_obs_keys(ob)

        with intervene_lock:
            must_skip_chunk = prev_step_intervened
            if prev_step_intervened:
                prev_step_intervened = False

        with action_queue_lock:
            need_refill = (len(action_queue) == 0)

        # 비어있고 intervene 직후면 dummy 동작 반환
        if need_refill and must_skip_chunk:
            action = np.zeros((action_dim,), dtype=np.float32)
            return {"action": action, "dummy": True}

        refill_chunk = None
        if need_refill:
            online_rng, key = jax.random.split(online_rng)
            with params_lock:
                a_chunk = agent.sample_actions(observations=processed_ob, rng=key)
            refill_chunk = np.array(a_chunk).reshape(-1, action_dim).tolist()

        # 다시 락을 잡고, queue가 비어있으면 refill한 뒤 pop
        with action_queue_lock:
            if len(action_queue) == 0 and refill_chunk is not None:
                action_queue.extend(refill_chunk)
            if len(action_queue) == 0:
                action = np.zeros((action_dim,), dtype=np.float32)
                return {"action": action, "dummy": True}
            action = np.asarray(action_queue.pop(0), dtype=np.float32)

        return {"action": action, "dummy": False}

    # Transition을 replay buffer에 추가
    def push_transition(payload):
        nonlocal logger, prev_step_intervened, log_step, rpc_call_count, local_version, agent
        ob         = payload["ob"]
        next_ob    = payload["next_ob"]
        reward     = float(payload["reward"])
        terminated = bool(payload["terminated"])
        truncated  = bool(payload["truncated"])
        info       = payload.get("info", {})
        action     = np.asarray(payload["action"], dtype=np.float32)
        state      = payload.get("state")

        done = terminated or truncated
        processed_ob = _process_obs_keys(ob)
        processed_next_ob = _process_obs_keys(next_ob)

        with rpc_lock:
            rpc_call_count += 1

        with log_lock:
            log_step += 1

        if rpc_call_count % 1000 == 0:
            print(f"[LOG] Current Step: {rpc_call_count}")

        # 파라미터 저장
        if FLAGS.save_interval > 0 and (rpc_call_count % FLAGS.save_interval == 0):
            flags_snapshot = dict(FLAGS.flag_values_dict())
            async_saver.request(
                agent=agent, step=log_step, params_ref=params_ref,
                rl_buffer=rl_buffer, demo_buffer=demo_buffer,
                log_step=log_step, rpc_call_count=rpc_call_count,
                version_id=version_id, flags_snapshot=flags_snapshot, online_rng=online_rng
            )

        # 파라미터 pull
        # if FLAGS.param_pull_interval > 0 and (rpc_call_count % FLAGS.param_pull_interval == 0) and (local_version != version_id['val']):
        #     with params_lock:
        #         agent = agent.replace(network=agent.network.replace(params=params_ref['params']))
        #         local_version = version_id['val']
        #         print(f"[LOG] Param pulled: Version {local_version}")
        #     with action_queue_lock:
        #         action_queue.clear()

        if "intervene_action" in info: 
            used_action = np.asarray(info.pop("intervene_action"), dtype=np.float32)
            intervened = True 
        else: 
            used_action = action.astype(np.float32, copy=False)
            intervened = False 
            
        if intervened and FLAGS.override_aborts_chunk:    
            with action_queue_lock:
                action_queue.clear()

        with intervene_lock:
            prev_step_intervened = intervened

        transition = dict(
            observations=processed_ob,
            actions=used_action,
            rewards=reward,
            terminals=float(done),
            masks=1.0 - float(terminated),
            next_observations=processed_next_ob,
            timeouts=float(done),
        )
        with rl_buffer_lock:
            rl_buffer.add_transition(transition)

        env_info = {k: v for k, v in info.items() if str(k).startswith("distance")}
        if env_info:
            logger.log(env_info, "env", step=log_step)

        if FLAGS.save_all_online_states and state is not None:
            i = payload.get("step_idx", None)  # 있으면 같이 저장
            if i is not None: data["steps"].append(int(i))
            if "qpos" in state: data["qpos"].append(np.asarray(state["qpos"]))
            if "qvel" in state: data["qvel"].append(np.asarray(state["qvel"]))
            if "button_states" in state: data["button_states"].append(np.asarray(state["button_states"]))
            data["obs"].append(processed_next_ob["state"] if "state" in processed_next_ob else np.nan)

        return {"ok": True}

    server.register_interface("on_reset", on_reset)
    server.register_interface("act", act)
    server.register_interface("push_transition", push_transition)

    print(f"[Trainer RPC] listening on :{FLAGS.trainer_rpc_port}")
    server.start()

    try:
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        learner_thread.join(timeout=2.0)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    end_time = time.time()

    if FLAGS.save_all_online_states and len(data["obs"]) > 0:
        c_data = {"online_time": end_time - online_init_time}
        if len(data["steps"]) > 0: c_data["steps"] = np.array(data["steps"])
        if len(data["qpos"])  > 0: c_data["qpos"]  = np.stack(data["qpos"], axis=0)
        if len(data["qvel"])  > 0: c_data["qvel"]  = np.stack(data["qvel"], axis=0)
        if len(data["obs"])   > 0 and isinstance(data["obs"][0], np.ndarray):
            c_data["obs"] = np.stack(data["obs"], axis=0)
        if len(data["button_states"]) > 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    flags_snapshot = dict(FLAGS.flag_values_dict())
    async_saver.flush_and_stop(
        agent=agent, step=log_step, params_ref=params_ref,
        rl_buffer=rl_buffer, demo_buffer=demo_buffer,
        log_step=log_step, rpc_call_count=rpc_call_count,
        version_id=version_id, flags_snapshot=flags_snapshot, online_rng=online_rng,
        timeout=3.0
    )

if __name__ == '__main__':
    app.run(main)