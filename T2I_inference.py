from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import time
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




lora_model_path = "/projects/AIGC/experiments2/rank_relax/"

pipe.load_lora_weights(lora_model_path, weight_name="pytorch_lora_weights.safetensors")

# pipe.load_lora_weights(lora_model_path)
prompt = "A [V] face"

negative_prompt = "nsfw,easynegative"
# negative_prompt = "nsfw, easynegative, paintings, sketches, (worst quality:2), low res, normal quality, ((monochrome)), skin spots, acnes, skin blemishes, extra fingers, fewer fingers, strange fingers, bad hand, mole, ((extra legs)), ((extra hands)), bad-hands-5"
# prompt = "1girl, stulmna, exquisitely detailed skin, looking at viewer, ultra high res, delicate"
# prompt = "A [v] face"
# prompt = "A pixarstyle of a [V] face"
# prompt = "A [V] face with bark skin"
# prompt = "A [V] face"
# prompt = "A professional portrait of a [V] face"
# prompt = "1girl, lineart, monochrome"

# prompt = "1girl,(exquisitely detailed skin:1.3), looking at viewer, ultra high res, delicate"
# prompt = "1boy, a professional detailed high-quality image, looking at viewer"
# prompt = "1girl, stulmno, solo, best quality, looking at viewer"
# prompt = "1girl, solo, best quality, looking at viewer"

# prompt = "(upper body: 1.5),(white background:1.4),  (illustration:1.1),(best quality),(masterpiece:1.1),(extremely detailed CG unity 8k wallpaper:1.1), (colorful:0.9),(panorama shot:1.4),(full body:1.05),(solo:1.2), (ink splashing),(color splashing),((watercolor)), clear sharp focus,{ 1boy standing },((chinese style )),(flowers,woods),outdoors,rocks, looking at viewer,  happy expression ,soft smile, detailed face, clothing decorative pattern details, black hair,black eyes, <lora:Colorwater_v4:0.8>"

pipe.to("cuda")

t0 = time.time()
for i in range(10):
    # image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=30, guidance_scale=7.5,cross_attention_kwargs={"scale":1}).images[0]
    image = pipe(prompt,  height=512, width=512, num_inference_steps=30).images[0]
    image.save("aigc_samples/test_%d.jpg" % i)
t1 = time.time()
print("time elapsed: %f"%((t1-t0)/10))
print("LoRA: %s"%lora_model_path)
