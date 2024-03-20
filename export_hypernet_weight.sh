export MODEL_NAME="stable-diffusion-models/realisticVisionV40_v40VAE"
export HYPER_WEIGHT_DIR="projects/AIGC/experiments2/hypernet/CelebA-HQ-10k-no-pretrain2"
export OUTPUT_DIR="projects/AIGC/lora_model_test"
export reference_image_path="dataset/FFHQ_test/00015.png"


python "export_hypernet_weight.py" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --hypernet_model_path $HYPER_WEIGHT_DIR \
  --output_dir $OUTPUT_DIR \
  --rank 1 \
  --down_dim 160 \
  --up_dim 80 \
  --train_text_encoder \
  --reference_image_path $reference_image_path
