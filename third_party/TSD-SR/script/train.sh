export MODEL_NAME="/path/to/your/sd3_model";
export TEACHER_MODEL_NAME="checkpoint/teacher/";
export CHECKPOINT_PATH="checkpoint/tsdsr";
export DEFAULT_EMBED="dataset/default"
export NULL_EMBED="dataset/null";
export HF_ENDPOINT="https://hf-mirror.com";
export OUTPUT_DIR="checkpoint/tsdsr-save/";
export OUTPUT_LOG="logs/tsdsr.log";
export LOG_NAME="tsdsr-train";
nohup accelerate launch  --config_file config/config.yaml  --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --main_process_port 57079 --mixed_precision="fp16" train/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --teacher_lora_path=$TEACHER_MODEL_NAME \
  --default_embedding_dir=$DEFAULT_EMBED --null_embedding_dir=$NULL_EMBED \
  --train_batch_size=2 --rank=64 --rank_vae=64 --rank_lora=64  \
  --num_train_epochs=200 --checkpointing_steps=5000 --validation_steps=500  --max_train_steps=200000 \
  --learning_rate=5e-06  --learning_rate_reg=1e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=3000 \
  --seed=43 --use_default_prompt --use_teacher_lora --use_random_bias \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" --log_code --log_name=$LOG_NAME \
  --gradient_accumulation_steps=1 \
  --resume_from_checkpoint="latest" \
  --guidance_scale=7.5  > $OUTPUT_LOG 2>&1 & \
