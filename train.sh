export XLA_FLAGS=--xla_gpu_autotune_level=0
export JAX_DEFAULT_matmul_precision='float32'
export XLA_PYTHON_CLIENT_PREALLOCATE=false 
export CUDA_VISIBLE_DEVICES=0

MUJOCO_GL=egl python main.py \
  --eval_interval 0 \
  --run_group=MY_ROBOT_HOR_10_MASKED \
  --env_name=AGI_TF \
  --sparse=True \
  --horizon_length 10 \
  --custom_dataset_dir ./preprocessed_datasets_state \
  --save_interval 10000 \
  --log_interval 1000 \
  --online_steps 100000 \
  --offline_steps 10000000 \
