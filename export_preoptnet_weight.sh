export MODEL_NAME="stable-diffusion-models/realisticVisionV40_v40VAE"
export PRE_OPTNET_WEIGHT_DIR="projects/AIGC/experiments/pretrained/CelebA-HQ-100"
export OUTPUT_DIR="projects/AIGC/lora_model_test"

python "export_preoptnet_weight.py" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --pre_opt_weight_path $PRE_OPTNET_WEIGHT_DIR \
  --output_dir $OUTPUT_DIR \
  --vit_model_name vit_huge_patch14_clip_336 \
  --rank 1 \
  --down_dim 160 \
  --up_dim 80 \
  --train_text_encoder \
  --total_identities 100 \
  --reference_image_id 10
