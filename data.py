import os
import random
import pickle

from tqdm import tqdm
import torch
import torch.nn.functional as F

from datasets import load_dataset

from diffusers import DPMSolverMultistepScheduler, Flux2KleinPipeline

from common import 计时, clean
from common import encode_prompt as encode_prompt_sdxl
from dan后处理 import dan后处理


def compute_text_embeddings(prompt, text_encoding_pipeline, max_sequence_length, text_encoder_out_layers):
    with torch.no_grad():
        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
    return prompt_embeds, text_ids


def compute_time_ids(original_size, resized_size, crops_coords_top_left):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = resized_size
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    return add_time_ids


def flux_vae_encode(vae, pixel_values, latents_bn_mean, latents_bn_std) -> torch.Tensor:
    latent = vae.encode(pixel_values.to(vae.dtype)).latent_dist.mode()
    latent = Flux2KleinPipeline._patchify_latents(latent)
    latent = (latent - latents_bn_mean) / latents_bn_std
    latent = Flux2KleinPipeline._unpatchify_latents(latent)
    return latent


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompts = [example["prompts"] for example in examples]
    prompts改 = [example["prompts改"] for example in examples]
    original_sizes = [example["original_sizes"] for example in examples]
    resized_sizes = [example["resized_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "prompts改": prompts改,
        "original_sizes": original_sizes,
        "resized_sizes": resized_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch


def 生成dataset(accelerator, train_data_dir, drop_tag_rate, drop_char_feature_rate, 学人rate, min_size, max_size) -> tuple:
    d后 = dan后处理(drop_tag_rate, drop_char_feature_rate, (min_size, max_size), 学人rate=学人rate)
    dataset = load_dataset(
        "imagefolder",
        data_files={"train": os.path.join(train_data_dir, "**")},
    )
    if accelerator:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(d后.preprocess_train, output_all_columns=True)
    else:
        train_dataset = dataset["train"].with_transform(d后.preprocess_train, output_all_columns=True)
    return train_dataset


def prefetch(it, accelerator, global_step, reform_prompt, prefetch_steps, drop_text_rate, teacher_cfg, inference_steps, text_encoder, 教师pipeline, text_encoding_pipeline, 教师pipeline的compel, vae, latents_bn_mean, latents_bn_std, max_sequence_length, text_encoder_out_layers, device) -> list[dict]:
    prefetch进度条 = tqdm(desc='prefetch', total=prefetch_steps)
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(教师pipeline.scheduler.config)
    batch_buffer = []
    with torch.no_grad():
        teacher_nega_prompt_embeds, teacher_nega_pooled_prompt_embeds = [i.cpu() for i in encode_prompt_sdxl([''], 教师pipeline的compel)]

        with 计时(accelerator, global_step, 'dataloader'):
            while len(batch_buffer) < prefetch_steps:
                batch = next(it)
                h, w = batch['pixel_values'].shape[2:]
                if h / w > 4 or w / h > 4:
                    continue
                batch_buffer.append(batch)
                if reform_prompt:
                    batch['teacher_prompts'] = batch['prompts'].copy()
                    batch['prompts'] = batch['prompts改'].copy()
                else:
                    batch['teacher_prompts'] = batch['prompts'].copy()
                batch['学nega'] = False
                if random.random() < drop_text_rate:
                    batch['学nega'] = True
                    batch['prompts'] = ['' for _ in batch['prompts']]
                    batch['sdxl_nega_embeds'] = teacher_nega_prompt_embeds

        with 计时(accelerator, global_step, 'TE_to_CUDA_1'):
            text_encoder.to(device)
        with 计时(accelerator, global_step, 'prepare_text_emb_1'):
            for batch in batch_buffer:
                batch['prompt_embeds'], batch['text_ids'] = [i.cpu() for i in compute_text_embeddings(batch['prompts'], text_encoding_pipeline, max_sequence_length, text_encoder_out_layers)]
        with 计时(accelerator, global_step, 'TE_to_CPU_1'):
            text_encoder.to('cpu')

        with 计时(accelerator, global_step, 'TE_to_CUDA_2'):
            for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                i.to(device)
        with 计时(accelerator, global_step, 'prepare_text_emb_2'):
            for i, batch in enumerate(batch_buffer):
                batch['sdxl_prompt_embeds'], batch['sdxl_pooled_prompt_embeds'] = [i.cpu() for i in encode_prompt_sdxl(batch['teacher_prompts'], 教师pipeline的compel)]
        with 计时(accelerator, global_step, 'TE_to_CPU_2'):
            for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                i.to('cpu')

        with 计时(accelerator, global_step, 'prepare_latent'):
            for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                i.to(device)
            for batch in batch_buffer:
                prefetch进度条.update(1)
                pixel_values = batch['pixel_values']
                bsz = pixel_values.shape[0]
                assert bsz == 1
                noise = torch.randn(size=(bsz, 32, pixel_values.shape[-2]//8, pixel_values.shape[-1]//8), dtype=torch.float32, device=vae.device)
                noise小 = noise.unflatten(1, (4, 8)).mean(dim=2) * (8**0.5)

                torch.cuda.empty_cache()

                assert teacher_cfg > 1
                to = dict(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype)
                add_time_ids = torch.cat([compute_time_ids(s, r, c) for s, r, c in zip(batch['original_sizes'], batch['resized_sizes'], batch['crop_top_lefts'])])
                zwei_add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(**to)
                cond_prompt_embeds = batch['sdxl_prompt_embeds']
                uncond_prompt_embeds = teacher_nega_prompt_embeds
                if uncond_prompt_embeds.shape[1] < cond_prompt_embeds.shape[1]:
                    uncond_prompt_embeds = F.pad(uncond_prompt_embeds, (0, 0, 0, cond_prompt_embeds.shape[1] - uncond_prompt_embeds.shape[1]))
                zwei_prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0).to(**to)
                zwei_pooled_prompt_embeds = torch.cat([teacher_nega_pooled_prompt_embeds, batch['sdxl_pooled_prompt_embeds']], dim=0).to(**to)
                latents = noise小.to(device=教师pipeline.unet.device, dtype=torch.float32)
                latents = latents * dpm_scheduler.init_noise_sigma
                dpm_scheduler.set_timesteps(inference_steps, device=教师pipeline.unet.device)
                for i, t in enumerate(dpm_scheduler.timesteps):
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    latent_model_input = dpm_scheduler.scale_model_input(latent_model_input, t)
                    with 计时(accelerator, global_step, 'unet'):
                        batched_pred = 教师pipeline.unet(
                            sample=latent_model_input.to(**to),
                            timestep=t,
                            encoder_hidden_states=zwei_prompt_embeds,
                            added_cond_kwargs={'time_ids': zwei_add_time_ids, 'text_embeds': zwei_pooled_prompt_embeds},
                            return_dict=False,
                        )[0]
                        batched_pred = batched_pred.to(torch.float32)
                        教师pred_uncond, 教师pred_cond = batched_pred.chunk(2, dim=0)
                        noise_pred = 教师pred_uncond + teacher_cfg * (教师pred_cond - 教师pred_uncond)
                        latents = dpm_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                教师pred_x0 = latents.detach().clone()
                latents_to_decode = 教师pred_x0 / 教师pipeline.vae.config.scaling_factor
                latents_to_decode = latents_to_decode.to(dtype=教师pipeline.vae.dtype)
                image_pixels = 教师pipeline.vae.decode(latents_to_decode, return_dict=False)[0]

                image_pixels = torch.clamp(image_pixels, min=-1.0, max=1.0)
                target_x0_sd3 = flux_vae_encode(vae, image_pixels, latents_bn_mean, latents_bn_std)
                batch['noise'] = noise.cpu()
                batch['target_x0_sd3'] = target_x0_sd3.cpu()
            del latents, 教师pred_x0, latents_to_decode, image_pixels, noise小, noise, zwei_prompt_embeds, zwei_pooled_prompt_embeds, zwei_add_time_ids
            for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                i.to('cpu')

        for batch in batch_buffer:
            del batch['pixel_values'], batch['sdxl_pooled_prompt_embeds']

        with 计时(accelerator, global_step, 'clean'):
            clean()

    return batch_buffer


# python data.py --all_ep=101,102 --train_data_dir=X:/image_balance大_2024 --pretrained_model_name_or_path="R:/models/FLUX.2-klein-base-4B" --teacher_model_name_or_path="S:/Stable-diffusion-models/Stable-diffusion/waiNSFWIllustrious_v140.safetensors" --output_dir="x:/RUM缓存_勇气2"
# python data.py --all_ep=204,205 --train_data_dir=X:/image_balance大_2024 --pretrained_model_name_or_path="C:/Users/Administrator/Desktop/FLUX.2-klein-base-4B" --teacher_model_name_or_path="C:/Users/Administrator/Desktop/models/waiNSFWIllustrious_v140.safetensors" --output_dir="x:/RUM缓存_勇气2"


def _ember(all_ep, train_data_dir, pretrained_model_name_or_path, teacher_model_name_or_path, output_dir):
    import time
    import pickle
    from diffusers import StableDiffusionXLPipeline, Flux2KleinPipeline
    from compel import Compel, ReturnedEmbeddingsType
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
    from diffusers import AutoencoderKLFlux2
    from accelerate.utils import set_seed
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(output_dir, exist_ok=True)
    set_seed(int(time.time()))

    学人rate = 0.5
    drop_tag_rate = 0.1
    drop_text_rate = 0.05
    drop_char_feature_rate = 0.6
    reform_prompt = False
    prefetch_steps = 200
    teacher_cfg = 6.5
    inference_steps = 16
    min_size = 576
    max_size = 1280

    tokenizer = Qwen2TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae"
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to('cuda:0')
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to('cuda:0')

    教师pipeline = StableDiffusionXLPipeline.from_single_file(teacher_model_name_or_path, torch_dtype=torch.float16)
    教师pipeline的compel = Compel(truncate_long_prompts=False, tokenizer=[教师pipeline.tokenizer, 教师pipeline.tokenizer_2], text_encoder=[教师pipeline.text_encoder, 教师pipeline.text_encoder_2],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    教师pipeline.vae.to(dtype=torch.float32)
    教师pipeline.unet.requires_grad_(False)
    教师pipeline.vae.requires_grad_(False)
    教师pipeline.text_encoder.requires_grad_(False)
    教师pipeline.text_encoder_2.requires_grad_(False)

    train_dataset = 生成dataset(None, train_data_dir, drop_tag_rate, drop_char_feature_rate, 学人rate, min_size, max_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        prefetch_factor=int(prefetch_steps * 1.1) + 1,
        num_workers=1,
    )
    pool = ThreadPoolExecutor(max_workers=1)

    def _dump到硬盘(a: list[dict], output_dir: str, ep):
        nonlocal i
        for d in a:
            d['prompt_embeds'] = d['prompt_embeds'].to(torch.bfloat16)
            with open(f'{output_dir}/ep{ep}_{i}.pkl', 'wb') as f:
                pickle.dump(d, f)
            i += 1

    for ep in all_ep:
        i = 0
        it = iter(train_dataloader)
        while True:
            try:
                a = prefetch(it, None, 0, reform_prompt, prefetch_steps, drop_text_rate, teacher_cfg, inference_steps, text_encoder, 教师pipeline, text_encoding_pipeline, 教师pipeline的compel, vae, latents_bn_mean, latents_bn_std, max_sequence_length=200, text_encoder_out_layers=[10, 20, 30], device='cuda:0')
                pool.submit(_dump到硬盘, a, output_dir, ep)
            except StopIteration:
                break


if __name__ == '__main__':
    import fire
    fire.Fire(_ember)
