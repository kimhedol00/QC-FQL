export XLA_FLAGS=--xla_gpu_autotune_level=0
export JAX_DEFAULT_matmul_precision='float32'
export XLA_PYTHON_CLIENT_PREALLOCATE=false 
export CUDA_VISIBLE_DEVICES=3

MUJOCO_GL=egl python main.py \
  --eval_interval 0 \
  --run_group=Hanoi \
  --env_name=AGI_TF \
  --sparse=True \
  --horizon_length 10 \
  --custom_dataset_dir ./hanoi/ \
  --save_interval 50000 \
  --log_interval 100 \
  --online_steps 0 \
  --offline_steps 100000 \
