from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import random
pretrain_model_path="stable-diffusion-models/realisticVisionV40_v40VAE"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1
)

pipe = StableDiffusionPipeline.from_pretrained(pretrain_model_path,
                                           torch_dtype=torch.float16,
                                           scheduler=noise_scheduler,
                                           requires_safety_checker=False)



# personal lora
dir1 = "projects/AIGC/lora_model_test"
lora_model_path1 = "pytorch_lora_weights.safetensors"
# prompt = "A [V] face"
dir2="projects/AIGC/experiments/style_lora"
lora_model_path2 = "pixarStyleModel_lora128.safetensors"
# lora_model_path2 = "Watercolor_Painting_by_vizsumit.safetensors"
# lora_model_path2 = "Professional_Portrait_1.5-000008.safetensors"
prompt = "A pixarstyle of a [V] face"
# prompt = "A watercolor paint of a [V] face"
# prompt = "A professional portrait of a [V] face"

negative_prompt = "nsfw,easynegative"

pipe.to("cuda")


pipe.load_lora_weights(dir1, weight_name=lora_model_path1, adapter_name="person")
pipe.load_lora_weights(dir2, weight_name=lora_model_path2, adapter_name="style")
# pipe.set_adapters(["person", "style"], adapter_weights=[0.6, 0.4])  #pixar

pipe.set_adapters(["person", "style"], adapter_weights=[0.4, 0.4])  #watercolor
# Fuses the LoRAs into the Unet
pipe.fuse_lora()

for i in range(10):
    seed = random.randint(0, 100)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5,cross_attention_kwargs={"scale":1}).images[0]
    # image = pipe(prompt,  height=512, width=512, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)).images[0]
    # image = pipe(prompt,  height=512, width=512, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}).images[0]
    image = pipe(prompt,  height=512, width=512, num_inference_steps=30, generator=generator).images[0]

    image.save("aigc_samples/test_after_export_%d.jpg" % i)

# Gets the Unet back to the original state
pipe.unfuse_lora()