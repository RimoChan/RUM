import os
import time
import copy
import math
import queue
import shutil
import random
import pickle
import logging
import platform
import argparse
import itertools
import threading
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torchvision.transforms.functional
from torchvision.transforms import InterpolationMode
import transformers
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers.utils.torch_utils
from diffusers import FlowMatchEulerDiscreteScheduler,  StableDiffusionXLPipeline, Flux2KleinPipeline, AutoencoderKLFlux2
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from compel import Compel, ReturnedEmbeddingsType

from common import 计时, 哈, cycle, clean, 生成optimizer, 评测pipeline, 评测pipeline人, add_image_jpeg, validation_prompt, validation_prompt_reform, edit_prompt, cosine_with_restart_scheduler改, optimizer_to_device, downsample_noise
from common import encode_prompt as encode_prompt_sdxl
from data import prefetch, 生成dataset, collate_fn, flux_vae_decode, flux_vae_encode
from arb import arb
from 哭 import 哭model

import muon
muon.zeropower_via_newtonschulz5 = torch.compile(muon.zeropower_via_newtonschulz5)


logger = get_logger(__name__)
diffusers.utils.torch_utils.logger.setLevel(logging.WARNING)


class 哭Pipeline(Flux2KleinPipeline):
    def 上床(self, prompt):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=args.max_sequence_length,
            text_encoder_out_layers=args.text_encoder_out_layers,
        )
        sdxl_prompt_embeds, _ = [i for i in encode_prompt_sdxl([prompt], self.教师pipeline的compel)]
        return torch.cat([prompt_embeds, F.pad(sdxl_prompt_embeds.to('cuda'), (0, 7680 - 2048))], dim=1).to(torch.bfloat16)

    @torch.inference_mode()
    def __call__(self, *, prompt, **kwargs):
        return super().__call__(prompt_embeds=self.上床(prompt), negative_prompt_embeds=self.上床(''), **kwargs)


def log_validation(
    pipeline,
    accelerator,
    global_step,
    guidance_scale=5,
    edit_guidance_scale=9,
):
    with torch.inference_mode():
        if args.edit_rate > 0:
            for num_inference_steps in [20]:
                images = []
                for prompt, seed in edit_prompt:
                    images.append(pipeline(
                        prompt=prompt,
                        image=Image.open("./img/抓人.jpg"),
                        generator=torch.Generator(device='cpu').manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=edit_guidance_scale,
                    ).images[0])
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        add_image_jpeg(tracker.writer, f"编辑-cfg{edit_guidance_scale}-n{num_inference_steps}", np.concatenate([np.asarray(img) for img in images], axis=1), global_step)
        for num_inference_steps in [20]:
            images = []
            for prompt, seed in validation_prompt:
                images.append(pipeline(
                    prompt=prompt,
                    generator=torch.Generator(device=accelerator.device).manual_seed(seed),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=704,
                    height=1024,
                ).images[0])
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    add_image_jpeg(tracker.writer, f"validation-cfg{guidance_scale}-n{num_inference_steps}", np.concatenate([np.asarray(img) for img in images], axis=1), global_step)
        clean()
        if global_step > 1188009:
            pipeline.set_progress_bar_config(disable=True)
            if 'human' in args.validation_type.split(','):
                分数 = 评测pipeline人(pipeline, n_iter=args.validation_n_iter, guidance_scale=guidance_scale)
                accelerator.log({f"人分数-{args.validation_n_iter}-cfg{guidance_scale}": 分数}, step=global_step)
            if 'tag' in args.validation_type.split(','):
                分数 = 评测pipeline(pipeline, n_iter=args.validation_n_iter//2, guidance_scale=guidance_scale)
                accelerator.log({f"tag分数-{args.validation_n_iter//2}-cfg{guidance_scale}": 分数}, step=global_step)
    del pipeline
    clean()


@torch.no_grad()
def 褪色(model, safetensors_path, alpha=0.99):
    beta = 1.0 - alpha
    total_diff = 0.0
    missing_keys = []

    b = load_file(safetensors_path)
    for name, param in model.named_parameters():
        if name not in b:
            missing_keys.append(name)
            continue
        orig_tensor = b[name].to(param.device)
        target_dtype = param.dtype
        current_fp32 = param.data.to(torch.float32)
        orig_fp32 = orig_tensor.to(torch.float32)
        total_diff += (param.numel() * torch.mean(torch.abs(current_fp32 - orig_fp32))).item()
        new_fp32 = (alpha * current_fp32) + (beta * orig_fp32)
        param.data.copy_(new_fp32.to(target_dtype))

    if missing_keys:
        print(f"这些参数找不到了: {missing_keys} ...")
    clean()
    return total_diff


@torch.no_grad()
def downsample_latent(vae, latent, latents_bn_mean, latents_bn_std, k):
    device = latent.device
    image = flux_vae_decode(vae, latent, latents_bn_mean, latents_bn_std)
    image = torch.clamp(image, min=-1.0, max=1.0)
    _, _, W, H = image.shape
    image = torchvision.transforms.functional.resize(
        image,
        size=[W // k, H // k], 
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    )
    image = torch.clamp(image, min=-1.0, max=1.0)
    latent = flux_vae_encode(vae, image, latents_bn_mean, latents_bn_std)
    return latent.to(device=device)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=200,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--validation_n_iter",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation_type",
        type=str,
        default='human',
    )
    parser.add_argument(
        "--prefetch_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--prefetch_n_sample",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prefetch_cache_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux_klein",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--fade_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume_transformer",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_muon",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--override_learning_rate",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_cosine_min",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--drop_text_rate",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--drop_sdxl_emb_rate",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--downsample_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--edit_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--drop_tag_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--drop_char_feature_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--sigmas_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--reform_prompt",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--offload_optimizer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--teacher_cfg",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--学人rate",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1344,
    )
    parser.add_argument(
        "--embedder_2_k",
        type=int,
        default=1,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    global validation_prompt
    if args.reform_prompt:
        validation_prompt = validation_prompt_reform
    if args.quick_test:
        validation_prompt = validation_prompt[:1]
        args.validation_steps = 6
        args.prefetch_steps = 5
        args.seed = 1
        args.validation_n_iter = 2

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is None:
        args.seed = int(time.time()) % 100
    set_seed(args.seed + int(os.environ.get('LOCAL_RANK', 0))*10000)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        accelerator.device
    )
    transformer = 哭model.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")

    transformer.context_embedder_2.to_empty(device="cpu")
    with torch.no_grad():
        torch.nn.init.normal_(transformer.context_embedder_2.weight, std=0.02)
        if hasattr(transformer.context_embedder_2, 'bias') and transformer.context_embedder_2.bias is not None:
            torch.nn.init.zeros_(transformer.context_embedder_2.bias)

    transformer.requires_grad_(True)
    vae.requires_grad_(False)

    assert accelerator.mixed_precision == "bf16", f'{accelerator.mixed_precision}不行，只支持bf16。'

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    教师pipeline = StableDiffusionXLPipeline.from_single_file(args.teacher_model_name_or_path, torch_dtype=torch.float16, **({'unet': None, 'vae': None} if args.prefetch_cache_dir else {}))
    教师pipeline的compel = Compel(truncate_long_prompts=False, tokenizer=[教师pipeline.tokenizer, 教师pipeline.tokenizer_2], text_encoder=[教师pipeline.text_encoder, 教师pipeline.text_encoder_2],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    if not args.prefetch_cache_dir:
        教师pipeline.vae.to(dtype=torch.float32)
        教师pipeline.unet.requires_grad_(False)
        教师pipeline.vae.requires_grad_(False)
    教师pipeline.text_encoder.requires_grad_(False)
    教师pipeline.text_encoder_2.requires_grad_(False)

    for k in ['vae', 'transformer', 'text_encoder', '教师pipeline.unet', '教师pipeline.vae', '教师pipeline.text_encoder', '教师pipeline.text_encoder_2']:
        v = eval(k)
        if v is not None:
            print(f'{k}({type(v).__name__})，参数量: {v.num_parameters(only_trainable=False) / 1e9:.2f} B，dtype={v.dtype}。')

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if not args.learning_rate_muon:
        lr_muon = args.learning_rate * 40
    elif args.learning_rate_muon > 1:
        lr_muon = args.learning_rate * args.learning_rate_muon
    else:
        lr_muon = args.learning_rate_muon
    optimizer = 生成optimizer(args.optimizer, transformer, args.adam_beta1, args.adam_beta2, args.adam_weight_decay, args.adam_epsilon, args.learning_rate, lr_muon, embedder_2_k=args.embedder_2_k, muon_weight_decay=args.muon_weight_decay)
    train_dataset = 生成dataset(accelerator, args.train_data_dir, args.drop_tag_rate, args.drop_char_feature_rate, args.学人rate, args.min_size, args.max_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        prefetch_factor=int(args.prefetch_steps*1.1) + 1,
        num_workers=args.dataloader_num_workers,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    assert args.lr_scheduler == 'cosine_with_restarts'
    lr_scheduler = cosine_with_restart_scheduler改(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        cosine_min=args.lr_cosine_min,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    特征 = f'{哈(args.train_data_dir)}-{哈(args.pretrained_model_name_or_path)}-{哈(args.teacher_model_name_or_path)}-{args.optimizer}-lr{args.learning_rate}-{args.lr_cosine_min}' + \
        f'b{args.train_batch_size}' * (args.train_batch_size > 1) + \
        f'x{args.gradient_accumulation_steps}' * (args.gradient_accumulation_steps > 1) + \
        f'x{world_size}' * (world_size > 1) + \
        f'-{args.lr_num_cycles}-drop{args.drop_text_rate}&{args.drop_tag_rate}&{args.drop_char_feature_rate}-{args.mixed_precision}-SS{args.sigmas_scale}-n{args.inference_steps}-cfg{args.teacher_cfg}-{args.weighting_scheme}_{args.logit_mean}_{args.logit_std}-down{args.downsample_rate}' + '-TEST'*bool(args.quick_test)
    if args.prefetch_n_sample != 1:
        特征 += f'-sp{args.prefetch_n_sample}'
    if args.max_grad_norm != 1:
        特征 += f'-norm{args.max_grad_norm}'
    if args.reform_prompt:
        特征 += f'-RP'
    if args.学人rate:
        特征 += f'-学人{args.学人rate}'
    if args.learning_rate_muon:
        特征 += f'-Muon{args.learning_rate_muon}'
    if args.muon_weight_decay != 1e-2:
        特征 += f'-D{args.muon_weight_decay}'
    特征 += f'-哭'
    if args.embedder_2_k != 1:
        特征 += f'{args.embedder_2_k}'

    if accelerator.is_main_process:
        args_cp = vars(args).copy()
        args_cp["text_encoder_out_layers"] = str(args_cp["text_encoder_out_layers"])
        accelerator.init_trackers(f'T2I-{特征}', config=args_cp)

    checkpoint_dir = os.path.join(args.output_dir, 特征)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    if args.resume_from_checkpoint == 'latest':
        if 候选checkpoint := [*Path(checkpoint_dir).glob('checkpoint-*')]:
            args.resume_from_checkpoint = str(max(候选checkpoint, key=lambda i: int(i.stem.split('-')[-1])))
        else:
            args.resume_from_checkpoint = None

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("-")[-1])
        initial_global_step = global_step
        if args.override_learning_rate:
            for group in optimizer.param_groups:
                if not group["use_muon"]:
                    group["lr"] = group["initial_lr"] = args.override_learning_rate
                else:
                    group["lr"] = group["initial_lr"] = args.override_learning_rate * 40
            lr_scheduler = cosine_with_restart_scheduler改(
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                last_epoch=global_step,
                cosine_min=args.lr_cosine_min,
            )
            lr_scheduler = accelerator.prepare(lr_scheduler)
    else:
        if args.resume_transformer:
            transformer.load_state_dict(load_file(args.resume_transformer))
            global_step = initial_global_step = 0
        else:
            global_step = initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def batch_n_sample(batch: dict, prefetch_n_sample:int = 1, 训练编辑=False) -> list[dict]:
        a = []
        标记 = random.randint(1000, 9999)
        for _ in range(prefetch_n_sample):
            b = copy.deepcopy(batch)
            if 训练编辑:
                b['noise'] = torch.randn(size=b['target_x0_sd3'].size(), dtype=b['target_x0_sd3'].dtype, device=b['target_x0_sd3'].device)
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=b['noise'].shape[0],
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=b['noise'].device)
            sigmas = get_sigmas(timesteps, n_dim=b['noise'].ndim, dtype=b['noise'].dtype).to(b['noise'].device)
            noisy_model_input = (1.0 - sigmas) * b['target_x0_sd3'] + sigmas * b['noise']
            b['timesteps'] = timesteps
            b['sigmas'] = sigmas
            b['noisy_model_input'] = Flux2KleinPipeline._patchify_latents(noisy_model_input)
            b['target_v'] = Flux2KleinPipeline._patchify_latents(b['noise'] - b['target_x0_sd3'])
            if 训练编辑:
                b['reference'] = Flux2KleinPipeline._patchify_latents(b['reference'])
            if not args.quick_test:
                del b['target_x0_sd3'], b['noise']
            b_cpu = {}
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    b_cpu[k] = v.cpu()
                else:
                    b_cpu[k] = v
            b_cpu['标记'] = 标记
            a.append(b_cpu)
        return a

    def 编源():
        while True:
            a = [*Path('S:/RUM_MagicBrush_去水印/').glob('*.pkl')]
            random.shuffle(a)
            for i in a:
                with open(i, 'rb') as f:
                    b = pickle.load(f)
                    yield from batch_n_sample(b, 1, True)

    def 超源(it, accelerator):
        if args.prefetch_cache_dir:
            q = queue.Queue(maxsize=2) 
            def _reader():
                while True:
                    a = [*Path(args.prefetch_cache_dir).glob('*.pkl')]
                    random.shuffle(a)
                    for i in a:
                        with open(i, 'rb') as f:
                            b = pickle.load(f)
                            s = b['prompts'][0]
                            if ('indoors' not in s) and ('outdoors' not in s) and random.random() < 0.5:
                                continue
                            q.put(b)
            threading.Thread(target=_reader, daemon=True).start()
            while True:
                batch = q.get(timeout=30)
                if random.random() < args.downsample_rate:
                    if random.random() < 0.5:
                        k = 2
                    else:
                        k = 4
                    with 计时(accelerator, global_step, '下采样'):
                        batch['noise'] = downsample_noise(batch['noise'], k)
                        batch['target_x0_sd3'] = downsample_latent(vae, batch['target_x0_sd3'], latents_bn_mean, latents_bn_std, k)
                yield from batch_n_sample(batch, 1)
        else:
            batch_buffer = []
            while True:
                if not batch_buffer:
                    transformer.to('cpu')
                    batch_buffer = prefetch(
                        it, accelerator, global_step, args.reform_prompt, args.prefetch_steps, args.drop_text_rate, args.teacher_cfg, args.inference_steps,
                        text_encoder, 教师pipeline, text_encoding_pipeline, 教师pipeline的compel, vae,
                        latents_bn_mean, latents_bn_std, args.max_sequence_length, args.text_encoder_out_layers,
                    )
                    with 计时(accelerator, global_step, 'sample_latent'):
                        batch_buffer_n = []
                        for batch in batch_buffer:
                            batch_buffer_n.extend(batch_n_sample(batch, args.prefetch_n_sample))
                        random.shuffle(batch_buffer_n)
                        batch_buffer = batch_buffer_n
                    with 计时(accelerator, global_step, 'clean'):
                        clean()
                    transformer.to(accelerator.device)
                else:
                    yield from batch_buffer
                    batch_buffer = []

    train_dataloader_超 = 超源(cycle(train_dataloader), accelerator)
    if args.train_batch_size > 1:
        train_dataloader_超 = arb(train_dataloader_超, args.train_batch_size)

    train_dataloader_编 = 编源()

    transformer.train()
    while global_step <= args.max_train_steps:
        if random.random() < args.edit_rate:
            训练编辑 = True
        else:
            训练编辑 = False
        if 训练编辑:
            batch = next(train_dataloader_编)
        else:
            batch = next(train_dataloader_超)
        with 计时(accelerator, global_step, 'step'):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                timesteps = batch['timesteps'].to(accelerator.device)
                noisy_model_input = batch['noisy_model_input'].to(accelerator.device)
                target = batch['target_v'].to(accelerator.device)
                sigmas = batch['sigmas'].to(accelerator.device)
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                if 训练编辑:
                    orig_input_shape = packed_noisy_model_input.shape
                    cond_model_input = batch['reference'].to(accelerator.device)
                    packed_noisy_model_input = torch.cat([
                        packed_noisy_model_input,
                        Flux2KleinPipeline._pack_latents(cond_model_input),
                    ], dim=1)

                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_model_input).to(device=accelerator.device)
                if 训练编辑:
                    orig_input_ids_shape = model_input_ids.shape
                    cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids([cond_model_input[0:1]]).to(device=cond_model_input.device)
                    cond_model_input_ids = cond_model_input_ids.expand(cond_model_input.shape[0], -1, -1)
                    model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(noisy_model_input.shape[0])
                else:
                    guidance = None

                if random.random() < args.drop_sdxl_emb_rate or 训练编辑:
                    超prompt_embeds = batch['prompt_embeds'].to(accelerator.device)
                else:
                    if batch['学nega']:
                        超prompt_embeds = torch.cat([batch['prompt_embeds'].to(accelerator.device), F.pad(batch['sdxl_nega_embeds'].to(accelerator.device), (0, 7680 - 2048))], dim=1)
                    else:
                        超prompt_embeds = torch.cat([batch['prompt_embeds'].to(accelerator.device), F.pad(batch['sdxl_prompt_embeds'].to(accelerator.device), (0, 7680 - 2048))], dim=1)
                bs, n = 超prompt_embeds.shape[0:2]
                超text_ids = torch.zeros(bs, n, 4, dtype=torch.long, device=accelerator.device)
                超text_ids[..., -1] = torch.arange(n, device=accelerator.device)

                model_pred_v = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=超prompt_embeds,
                    txt_ids=超text_ids,
                    img_ids=model_input_ids,
                    return_dict=False,
                )[0]
                if 训练编辑:
                    model_pred_v = model_pred_v[:, : orig_input_shape[1], :]
                    model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]
                else:
                    model_pred_v = model_pred_v[:, : packed_noisy_model_input.size(1) :]
                model_pred_v = Flux2KleinPipeline._unpack_latents_with_ids(model_pred_v, model_input_ids)

                if args.quick_test:
                    model_pred_x0 = model_pred_v * (-sigmas) + noisy_model_input
                    with torch.no_grad():
                        transformer.to('cpu')
                        存档文件夹 = f'{args.output_dir}/quick_test_image_cfg{args.teacher_cfg}_n{args.inference_steps}'
                        os.makedirs(存档文件夹, exist_ok=True)
                        for latent, 名字, use_patchify in [(model_pred_x0, '学生pred_x0', False), (noisy_model_input, 'xt', False), (batch['target_x0_sd3'], '教师x0', True)]:
                            if use_patchify:
                                latent = Flux2KleinPipeline._patchify_latents(latent)
                            latent = latent.to(accelerator.device) * latents_bn_std + latents_bn_mean
                            image = vae.decode(Flux2KleinPipeline._unpatchify_latents(latent).to(device=vae.device, dtype=vae.dtype), return_dict=False)[0]
                            image = text_encoding_pipeline.image_processor.postprocess(image, output_type='pil')[0]
                            image.save(f'{存档文件夹}/step{global_step}_{int(timesteps[0])}_{名字}.png')
                        transformer.to(accelerator.device)

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred_v.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                del model_pred_v, target, packed_noisy_model_input, model_input_ids, 超prompt_embeds, 超text_ids, noisy_model_input, sigmas, weighting
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                with 计时(accelerator, global_step, 'optimizer'):
                    if args.offload_optimizer and accelerator.sync_gradients:
                        optimizer_to_device(optimizer, accelerator.device)
                    if global_step % 5 == 0:
                        torch.cuda.empty_cache()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if args.offload_optimizer and accelerator.sync_gradients:
                        optimizer_to_device(optimizer, 'cpu')

        if accelerator.is_main_process and accelerator.sync_gradients and (global_step % args.validation_steps == 0 or global_step in [args.validation_steps // 2, args.validation_steps // 4]):
            with 计时(accelerator, global_step, 'validation'):
                optimizer_to_device(optimizer, 'cpu')
                text_encoder.to(accelerator.device)
                transformer.to(accelerator.device)

                for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                    i.to(accelerator.device)

                pipeline = 哭Pipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=accelerator.unwrap_model(transformer),
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                )
                pipeline.教师pipeline的compel = 教师pipeline的compel
                
                clean()
                log_validation(pipeline, accelerator, global_step, 5)

                for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                    i.to('cpu')

                transformer.to(accelerator.device)
                text_encoder.to('cpu')
                optimizer_to_device(optimizer, accelerator.device)
                clean()

        if accelerator.sync_gradients:
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm.item()}
            progress_bar.set_postfix(**logs)
            for a, b in itertools.pairwise([1000, 900, 800, 0]):
                if a >= timesteps[0] >= b:
                    logs = logs | {
                        f'grad_norm_{a}到{b}': logs['grad_norm'],
                        f'loss_{a}到{b}': logs['loss'],
                    }
                    accelerator.log(logs, step=global_step)
                    break
            progress_bar.update(1)
            accelerator.log({"len_tag": batch.get('prompts', '没有')[0].count(','), "t": timesteps[0]}, step=global_step)
            global_step += 1
            if args.fade_steps > 0 and global_step % args.fade_steps == 2:
                clean()
                权重diff = 褪色(transformer, args.pretrained_model_name_or_path + '/transformer/diffusion_pytorch_model.safetensors')
                accelerator.log({"权重diff": 权重diff}, step=global_step)
            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                if args.checkpoints_total_limit is not None and os.path.exists(checkpoint_dir):
                    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(checkpoint_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)
                save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                # windows上有内存泄漏。不过我不确定是不是windows的问题，总之先这样屏蔽1下吧。
                if platform.system() == 'Windows' and global_step % 40000 == 0:
                    exit()

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
