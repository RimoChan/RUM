import os
os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'

import json
import pickle
from pathlib import Path
from imgutils.tagging import get_wd14_tags

from tqdm import tqdm
import torch
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2


输入文件夹 = "S:/RUM缓存_花月_新"

所有人 = set(json.loads(open('./人频率6000000~7000000.json').read())) | set(json.loads(open('./人频率1~6400000.json').read()))
所有人 = {i.replace(' ', '_') for i in 所有人}

pipeline = Flux2KleinPipeline.from_pretrained(
    "R:/models/FLUX.2-klein-base-4B",
    vae=None,
    transformer=None,
    tokenizer=None,
    text_encoder=None,
    scheduler=None,
)
vae = AutoencoderKLFlux2.from_pretrained(
    "R:/models/FLUX.2-klein-base-4B",
    subfolder="vae"
)
vae.to('cuda:0')
latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to('cuda:0')
latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to('cuda:0')

好 = 坏 = 0
with torch.inference_mode():
    t = [*Path(输入文件夹).glob('*.pkl')]
    for i in tqdm(t):
        with open(i, 'rb') as f:
            batch = pickle.load(f)
        latent = batch['target_x0_sd3'].to('cuda:0')
        latent = Flux2KleinPipeline._patchify_latents(latent)
        latent = latent * latents_bn_std + latents_bn_mean
        latent = Flux2KleinPipeline._unpatchify_latents(latent)
        image = vae.decode(latent.to(device=vae.device, dtype=vae.dtype), return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type='pil')[0]
        预测 = set(get_wd14_tags(image, character_threshold=0.9)[2])
        教师tags = [i.replace(' ', '_') for i in batch['teacher_prompts'][0].split(', ')]
        应当角色 = set(教师tags) & 所有人
        if len(应当角色 & 预测) >= 2:
            好 += 1
        else:
            坏 += 1
            os.remove(i)
        print(好, 坏)
