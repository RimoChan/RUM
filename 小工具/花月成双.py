import os
import time
import subprocess
os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'

import json
import pickle
from pathlib import Path
from imgutils.tagging import get_wd14_tags

import fire
import torch
from tqdm import tqdm
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2


所有人 = {i.replace(' ', '_') for i in (set(json.loads(open('../人频率6000000~7000000.json').read())) | set(json.loads(open('../人频率1~6400000.json').read())))}
所有老人 = {i.replace(' ', '_') for i in set(json.loads(open('../人频率1~6400000.json').read()))}


坏标签 = {'monochrome', 'greyscale', 'multiple_views', 'speech_bubble', 'thought_bubble', 'traditional_media', 'border', 'column_lineup', '2koma', '3koma', '4koma', 'extra_arms', 'head_out_of_frame', '4girls', '4boys', 'comic', 'cover_page', 'dated', 'copyright_name', 'patreon_username', 'character_name', 'twitter_username', 'artist_name'}
多人标签 = {'2girls', '2boys', '3girls', '3boys'}


def 坏吗(batch, image):
    width, height = image.size
    if (width+height)/2 < 700:
        return True, '小了', 预测
    预测 = get_wd14_tags(image, character_threshold=0.9, general_threshold=0.3, model_name='EVA02_Large')
    预测标签 = set(预测[1])
    坏交 = 预测标签 & 坏标签
    if 坏交:
        return True, ",".join(坏交), 预测
    教师tags = [i.replace(' ', '_') for i in batch['teacher_prompts'][0].split(', ')]
    预测角色 = set(预测[2])
    if len(set(教师tags) & 所有老人) > 0 and len(预测角色) == 0:
        return True, '角色丢失', 预测
    if 多人标签 & set(教师tags):
        if len(set(教师tags) & 所有人 & 预测角色) < 2:
            return True, '多人图缺人', 预测
    return False, '', 预测


def 等上个任务结束(target_gb=2.0, 检查间隔=120):
    while True:
        used_gb = int(subprocess.check_output(["nvidia-smi", f"--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], encoding="utf-8").strip()) / 1024
        if used_gb < target_gb:
            break
        print(f"当前显存占用: {used_gb:.2f} GB，休息1下。")
        time.sleep(检查间隔)


# python 花月成双.py --输入文件夹="S:\RUM缓存_花月_新" --cuda_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
# python 花月成双.py --输入文件夹="S:\RUM缓存_花月_时间结界" --cuda_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
# python 花月成双.py --输入文件夹="S:\RUM缓存_谨慎水月_时间结界" --cuda_dir="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
# python 花月成双.py --输入文件夹="R:\RUM缓存_正义"
# python 花月成双.py --输入文件夹="R:\RUM缓存_正义2"
def ember(输入文件夹: str, 逆=False, cuda_dir=None):
    等上个任务结束()
    if cuda_dir:
        import onnxruntime
        onnxruntime.preload_dlls(cuda=True, cudnn=False, directory=cuda_dir)

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
    看看文件夹 = 输入文件夹 + "_删除图"
    os.makedirs(看看文件夹, exist_ok=True)
    with torch.inference_mode():
        t = sorted([*Path(输入文件夹).glob('*.pkl')])
        if 逆:
            t = t[::-1]
        for i in tqdm(t, ncols=60):
            if not i.exists():
                continue
            with open(i, 'rb') as f:
                batch = pickle.load(f)
            latent = batch['target_x0_sd3'].to('cuda:0')
            latent = Flux2KleinPipeline._patchify_latents(latent)
            latent = latent * latents_bn_std + latents_bn_mean
            latent = Flux2KleinPipeline._unpatchify_latents(latent)
            image = vae.decode(latent.to(device=vae.device, dtype=vae.dtype), return_dict=False)[0]
            image = pipeline.image_processor.postprocess(image, output_type='pil')[0]
            坏b, 原因, 预测 = 坏吗(batch, image)
            if 坏b:
                坏 += 1
                image.save(f'{看看文件夹}/{i.name}_{原因}.jpg')
                with open(f'{看看文件夹}/{i.name}_{原因}.txt', 'w') as f:
                    f.write(str(batch['teacher_prompts']))
                    f.write(str(预测))
                os.remove(i)
            else:
                好 += 1
            print(好, 坏)


if __name__ == '__main__':
    fire.Fire(ember)
