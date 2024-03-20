# Model
export MODEL_NAME="stable-diffusion-models/realisticVisionV40_v40VAE"
export PRE_OPT_WEIGHT_DIR="projects/AIGC/experiments2/pretrained/CelebA-HQ-10k-fake"
#vit_base_patch16_224
# Image
export INSTANCE_DIR="projects/AIGC/dataset/CelebA-HQ-10k"
export VALIDATION_INPUT_DIR="projects/AIGC/dataset/FFHQ_test"
# Output
export VALIDATION_OUTPUT_DIR="projects/AIGC/experiments2/validation_outputs_10k-4"
export OUTPUT_DIR="experiments/hypernet/CelebA-HQ-10k"


CUDA_VISIBLE_DEVICES=0 \
accelerate launch --mixed_precision="fp16" train_hypernet.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --pre_opt_weight_path $PRE_OPT_WEIGHT_DIR \
  --instance_data_dir $INSTANCE_DIR \
  --vit_model_name vit_huge_patch14_clip_336 \
  --instance_prompt "A [V] face" \
  --output_dir $OUTPUT_DIR \
  --allow_tf32 \
  --resolution 512 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --checkpoints_total_limit 10 \
  --checkpointing_steps 10000 \
  --cfg_drop_rate 0.1 \
  --seed=42 \
  --rank 1 \
  --down_dim 160 \
  --up_dim 80 \
  --train_batch_size 4 \
  --pre_opt_weight_coeff 0.02 \
  --num_train_epochs 400 \
  --resume_from_checkpoint "latest" \
  --validation_prompt="A [V] face" \
  --validation_input_dir $VALIDATION_INPUT_DIR \
  --validation_output_dir $VALIDATION_OUTPUT_DIR \
  --validation_epochs 10 \
  --train_text_encoder


# when train_text_encoder, not pre_compute_text_embeddings
#  --pre_compute_text_embeddings \
#  --train_text_encoder
#  --patch_mlp \


