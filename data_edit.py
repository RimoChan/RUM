import io
import os
import random
import pickle
from pathlib import Path

os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync,expandable_segments:True'

from tqdm import tqdm
import torch
from PIL import Image
import pyarrow.parquet as pq
from diffusers import AutoencoderKLFlux2
from diffusers import Flux2KleinPipeline
import torchvision.transforms as transforms
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from data import flux_vae_encode, compute_text_embeddings
from common import 计时, clean


from torchvision.transforms.functional import pil_to_tensor


def parquet_data_iterator(file_path, min_size=256, max_size=1087, tolerance=50.0):
    columns_to_read = ['img_id', 'turn_index', 'source_img', 'instruction', 'target_img']

    parquet_file = pq.ParquetFile(file_path)

    def decode_to_pil(img_data: dict):
        return Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")

    def resize_img(img: Image.Image, size: int):
        width, height = img.size
        short_edge = min(width, height)

        if short_edge != size:
            scale = size / short_edge
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        return img

    计数 = 0
    for batch in parquet_file.iter_batches(batch_size=100, columns=columns_to_read):
        batch_dict = batch.to_pydict()
        num_rows = len(batch_dict['img_id'])
        for i in range(num_rows):
            if batch_dict['turn_index'][i] > 1:
                continue
            计数 += 1
            source_img = decode_to_pil(batch_dict['source_img'][i])
            target_img = decode_to_pil(batch_dict['target_img'][i])

            s_width, s_height = source_img.size
            t_width, t_height = target_img.size

            chk_w, chk_h = 90, 26
            wm_w, wm_h = 80, 16

            src_chk_w = int(chk_w * (s_width / t_width))
            src_chk_h = int(chk_h * (s_height / t_height))

            target_chk_patch = target_img.crop((t_width - chk_w, t_height - chk_h, t_width, t_height))
            source_chk_patch = source_img.crop((s_width - src_chk_w, s_height - src_chk_h, s_width, s_height))

            source_chk_patch = source_chk_patch.resize((chk_w, chk_h), resample=Image.Resampling.LANCZOS)

            t_tensor = pil_to_tensor(target_chk_patch).float()
            s_tensor = pil_to_tensor(source_chk_patch).float()

            mask = torch.ones((3, chk_h, chk_w), dtype=torch.float32)
            mask[:, chk_h - wm_h:, chk_w - wm_w:] = 0.0  

            diff = torch.abs(t_tensor - s_tensor) * mask
            valid_pixels = torch.sum(mask)
            mae = torch.sum(diff) / valid_pixels

            if mae.item() > tolerance:
                print(f"{Path(file_path).name}_{计数-1} 背景不一致，跳过！ (MAE: {mae.item():.2f} > {tolerance})")
                continue

            source_wm_patch = source_chk_patch.crop((chk_w - wm_w, chk_h - wm_h, chk_w, chk_h))
            target_img.paste(source_wm_patch, (t_width - wm_w, t_height - wm_h))
            size = random.randint(min_size, max_size) // 64 * 64
            yield {
                'instruction': batch_dict['instruction'][i],
                'source_img': resize_img(source_img, size),
                'target_img': resize_img(target_img, size),
            }


def parquet_data_iterator_大(all_parquet):
    for p in all_parquet:
        for i, d in tqdm(enumerate(parquet_data_iterator(p)), desc=p.name[:11]):
            yield i, d, p.name


def _ember(magic_brush_data_dir: str, output_dir: str, pretrained_model_name_or_path: str, ):
    os.makedirs(output_dir, exist_ok=True)

    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )
    vae.to('cuda:0')
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to('cuda:0')
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to('cuda:0')

    tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to('cuda:0')
    text_encoder.requires_grad_(False)
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    pil转tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print(vae.device, text_encoder.device)

    with torch.inference_mode():
        all_parquet = Path(magic_brush_data_dir).glob('*.parquet')
        for i, d, p_name in parquet_data_iterator_大(all_parquet):
            if os.path.exists(f'{output_dir}/{p_name}_{i}.pkl'):
                continue
            d['reference'] = flux_vae_encode(vae, pil转tensor(d.pop('source_img')).unsqueeze(0), latents_bn_mean, latents_bn_std)
            d['target_x0_sd3'] = flux_vae_encode(vae, pil转tensor(d.pop('target_img')).unsqueeze(0), latents_bn_mean, latents_bn_std)
            d['prompt_embeds'], d['text_ids'] = [i.cpu() for i in compute_text_embeddings(d['instruction'], text_encoding_pipeline, 200, [10, 20, 30])]
            d['prompt_embeds'] = d['prompt_embeds'].to(torch.bfloat16)
            with open(f'{output_dir}/{p_name}_{i}.pkl', 'wb') as f:
                pickle.dump(d, f)
            if i % 10 == 0:
                clean()


# python data_edit.py --magic_brush_data_dir=S:/MagicBrush/data --output_dir=S:\RUM_MagicBrush_去水印 --pretrained_model_name_or_path="C:/Users/Administrator/Desktop/FLUX.2-klein-base-4B"


if __name__ == '__main__':
    import fire
    fire.Fire(_ember)
