export MODEL_NAME="stable-diffusion-models/realisticVisionV40_v40VAE"
export INSTANCE_DIR="projects/AIGC/dataset/AI_drawing/instance_dir"
export OUTPUT_DIR="projects/AIGC/lora_model_test"

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --mixed_precision="fp16" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="A [V] face" \
  --resolution=512  \
  --train_batch_size=1 \
  --num_train_epochs=201 --checkpointing_steps=500 \
  --learning_rate=5e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=5 \
  --validation_prompt="A [V] face" \
  --validation_epochs=200 \
  --train_text_encoder
#  --patch_mlp \
#  --resume_from_checkpoint=$RESUME_DIR \


