export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"

accelerate launch train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name="jasonchoi3/nordstrom96568" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=30 --checkpointing_steps=5000 \
  --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="outputs" --mixed_precision="fp16" \
  --report_to="wandb" \
  --local_rank=0