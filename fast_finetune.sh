export MODEL_NAME="stable-diffusion-models/realisticVisionV40_v40VAE"
INSTANCE_IMAGE_PATH="projects/AIGC/dataset/FFHQ_test/00083.png"

export RESUME_DIR="projects/AIGC/experiments2/rank_relax"
export OUTPUT_DIR="projects/AIGC/experiments2/fast_finetune"


CUDA_VISIBLE_DEVICES=0 \
accelerate launch --mixed_precision="fp16" fast_finetune.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_image_path=$INSTANCE_IMAGE_PATH \
  --instance_prompt="A [V] face" \
  --resolution=512  \
  --train_batch_size=1 \
  --num_train_steps=25 \
  --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --rank=4 \
  --output_dir=$OUTPUT_DIR \
  --resume_dir=$RESUME_DIR \
  --num_validation_images=5 \
  --validation_prompt="A [V] face" \
  --train_text_encoder
#  --patch_mlp \
#  --resume_from_checkpoint=$RESUME_DIR \


