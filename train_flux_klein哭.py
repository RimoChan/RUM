import os
import time
import copy
import math
import shutil
import random
import logging
import platform
import argparse
from pathlib import Path

from datasets import load_dataset
import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler,  StableDiffusionXLPipeline, Flux2KleinPipeline, AutoencoderKLFlux2
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from compel import Compel, ReturnedEmbeddingsType

from dan后处理 import dan后处理
from common import 计时, 哈, cycle, clean, sdxl_time_to_alpha, 生成optimizer, 评测pipeline, 评测pipeline人, add_image_jpeg, validation_prompt, validation_prompt_reform, cosine_with_restart_scheduler改, optimizer_to_device
from common import encode_prompt as encode_prompt_sdxl

from 哭 import 哭model

import muon
muon.zeropower_via_newtonschulz5 = torch.compile(muon.zeropower_via_newtonschulz5)


logger = get_logger(__name__)


class 哭Pipeline(Flux2KleinPipeline):
    @torch.no_grad()
    def __call__(self, *, prompt, **kwargs):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=args.max_sequence_length,
            text_encoder_out_layers=args.text_encoder_out_layers,
        )
        sdxl_prompt_embeds, _ = [i for i in encode_prompt_sdxl([prompt], self.教师pipeline的compel)]
        超prompt_embeds = torch.cat([prompt_embeds, F.pad(sdxl_prompt_embeds, (0, 7680 - 2048))], dim=1)
        return super().__call__(prompt_embeds=超prompt_embeds, **kwargs)


def log_validation(
    pipeline,
    accelerator,
    global_step,
    guidance_scale=5,
):
    images = []
    with torch.inference_mode():
        for prompt, seed in validation_prompt:
            images.append(pipeline(
                prompt=prompt,
                generator=torch.Generator(device=accelerator.device).manual_seed(seed),
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                width=704,
                height=1024,
            ).images[0])
        clean()
        if global_step > 0:
            pipeline.set_progress_bar_config(disable=True)
            if args.validation_type == 'human':
                分数 = 评测pipeline人(pipeline, n_iter=args.validation_n_iter, guidance_scale=guidance_scale)
                前缀 = '人分数'
            else:
                分数 = 评测pipeline(pipeline, n_iter=args.validation_n_iter, guidance_scale=guidance_scale)
                前缀 = '分数'
            accelerator.log({f"{前缀}-cfg{guidance_scale}": 分数}, step=global_step)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            concat_image = np.concatenate([np.asarray(img) for img in images], axis=1)
            add_image_jpeg(tracker.writer, f"validation-cfg{guidance_scale}", concat_image, global_step)
    del pipeline
    clean()
    return images


def 生成dataset(accelerator, drop_tag_rate, drop_char_feature_rate, 学人rate) -> tuple:
    d后 = dan后处理(drop_tag_rate, drop_char_feature_rate, (args.min_size, args.max_size), 学人rate=学人rate)
    dataset = load_dataset(
        "imagefolder",
        data_files={"train": os.path.join(args.train_data_dir, "**")},
    )
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(d后.preprocess_train, output_all_columns=True)
    return train_dataset


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
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
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
        "--output_dir",
        type=str,
        default="flux_klein",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
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
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--muon_weight_decay", type=float, default=1e-02)
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

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
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--drop_text_rate",
        type=float,
        default=0.02,
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


def compute_time_ids(original_size, resized_size, crops_coords_top_left):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = resized_size
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    return add_time_ids


def main(args):
    global validation_prompt
    if args.reform_prompt:
        validation_prompt = validation_prompt_reform
    if args.quick_test:
        validation_prompt = validation_prompt[:1]
        args.validation_steps = 6
        args.prefetch_steps = 3
        args.seed = 1
        args.validation_n_iter = 2

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

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

    # vae.to(dtype=torch.bfloat16)
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    教师pipeline = StableDiffusionXLPipeline.from_single_file(args.teacher_model_name_or_path, torch_dtype=torch.float16)
    教师pipeline的compel = Compel(truncate_long_prompts=False, tokenizer=[教师pipeline.tokenizer, 教师pipeline.tokenizer_2], text_encoder=[教师pipeline.text_encoder, 教师pipeline.text_encoder_2],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    教师pipeline.vae.to(dtype=torch.float32)
    教师pipeline.unet.requires_grad_(False)
    教师pipeline.vae.requires_grad_(False)
    教师pipeline.text_encoder.requires_grad_(False)
    教师pipeline.text_encoder_2.requires_grad_(False)

    for k in ['vae', 'transformer', 'text_encoder', '教师pipeline.unet', '教师pipeline.vae', '教师pipeline.text_encoder', '教师pipeline.text_encoder_2']:
        v = eval(k)
        print(f'{k}({type(v).__name__})参数量: {v.num_parameters(only_trainable=False) / 1e9:.2f} B')

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if not args.learning_rate_muon:
        lr_muon = args.learning_rate * 40
    elif args.learning_rate_muon > 1:
        lr_muon = args.learning_rate * args.learning_rate_muon
    else:
        lr_muon = args.learning_rate_muon
    optimizer = 生成optimizer(args.optimizer, transformer, args.adam_beta1, args.adam_beta2, args.adam_weight_decay, args.adam_epsilon, args.learning_rate, lr_muon, embedder_2_k=args.embedder_2_k, muon_weight_decay=args.muon_weight_decay)
    train_dataset = 生成dataset(accelerator, args.drop_tag_rate, args.drop_char_feature_rate, args.学人rate)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        prefetch_factor=int(args.prefetch_steps*1.1) + 1,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=args.text_encoder_out_layers,
            )
        return prompt_embeds, text_ids

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
        f'x{args.gradient_accumulation_steps}' * (args.gradient_accumulation_steps > 1) + \
        f'x{world_size}' * (world_size > 1) + \
        f'-{args.lr_num_cycles}-drop{args.drop_text_rate}&{args.drop_tag_rate}&{args.drop_char_feature_rate}-{args.mixed_precision}-SS{args.sigmas_scale}-n{args.inference_steps}-cfg{args.teacher_cfg}-{args.weighting_scheme}_{args.logit_mean}_{args.logit_std}' + '-TEST'*bool(args.quick_test)
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
            from safetensors.torch import load_file
            state_dict = load_file(args.resume_transformer)
            transformer.load_state_dict(state_dict)
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

    def flux_vae_encode(vae, pixel_values) -> torch.Tensor:
        latent = vae.encode(pixel_values.to(vae.dtype)).latent_dist.mode()
        latent = Flux2KleinPipeline._patchify_latents(latent)
        latent = (latent - latents_bn_mean) / latents_bn_std
        latent = Flux2KleinPipeline._unpatchify_latents(latent)
        return latent

    def 超源(it, accelerator):
        batch_buffer = []
        teacher_nega_prompt_embeds, teacher_nega_pooled_prompt_embeds = [i.cpu() for i in encode_prompt_sdxl([''], 教师pipeline的compel)]
        while True:
            if not batch_buffer:
                transformer.to('cpu')
                with torch.no_grad():
                    with 计时(accelerator, global_step, 'dataloader'):
                        while len(batch_buffer) < args.prefetch_steps:
                            batch = next(it)
                            h, w = batch["pixel_values"].shape[2:]
                            if h / w > 4 or w / h > 4:
                                continue
                            batch_buffer.append(batch)
                            if args.reform_prompt:
                                batch['teacher_prompts'] = batch['prompts'].copy()
                                batch['prompts'] = batch['prompts改'].copy()
                            else:
                                batch['teacher_prompts'] = batch['prompts'].copy()
                            if random.random() < args.drop_text_rate:
                                batch['prompts'] = ['' for _ in batch['prompts']]

                    with 计时(accelerator, global_step, 'TE_to_CUDA_1'):
                        text_encoder.to(accelerator.device)
                    with 计时(accelerator, global_step, 'prepare_text_emb_1'):
                        for batch in batch_buffer:
                            batch['prompt_embeds'], batch['text_ids'] = [i.cpu() for i in compute_text_embeddings(batch['prompts'], text_encoding_pipeline)]
                    with 计时(accelerator, global_step, 'TE_to_CPU_1'):
                        text_encoder.to('cpu')

                    with 计时(accelerator, global_step, 'TE_to_CUDA_2'):
                        for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                            i.to(accelerator.device)
                    with 计时(accelerator, global_step, 'prepare_text_emb_2'):
                        for i, batch in enumerate(batch_buffer):
                            batch['sdxl_prompt_embeds'], batch['sdxl_pooled_prompt_embeds'] = [i.cpu() for i in encode_prompt_sdxl(batch['teacher_prompts'], 教师pipeline的compel)]
                    with 计时(accelerator, global_step, 'TE_to_CPU_2'):
                        for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                            i.to('cpu')

                    with 计时(accelerator, global_step, 'prepare_latent'):
                        for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                            i.to(accelerator.device)
                        for batch in batch_buffer:
                            pixel_values = batch["pixel_values"]
                            bsz = pixel_values.shape[0]
                            assert bsz == 1
                            noise = torch.randn(size=(bsz, 32, pixel_values.shape[-2]//8, pixel_values.shape[-1]//8), dtype=torch.float32, device=vae.device)
                            noise小 = noise.unflatten(1, (4, 8)).mean(dim=2) * (8**0.5)

                            u = compute_density_for_timestep_sampling(
                                weighting_scheme=args.weighting_scheme,
                                batch_size=bsz,
                                logit_mean=args.logit_mean,
                                logit_std=args.logit_std,
                                mode_scale=args.mode_scale,
                            )
                            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                            timesteps = noise_scheduler_copy.timesteps[indices].to(device=noise.device)

                            sigmas = get_sigmas(timesteps, n_dim=noise.ndim, dtype=noise.dtype)

                            torch.cuda.empty_cache()

                            初始beta = 1 - 0.02
                            for i in range(args.inference_steps):
                                beta = 初始beta * (1 - i / args.inference_steps)
                                alpha = 1 - beta
                                if i == 0:
                                    noisy_model_input小 = noise小
                                else:
                                    noisy_model_input小 = alpha**0.5 * 教师pred_x0 + beta**0.5 * 教师pred
                                with 计时(accelerator, global_step, 'unet'):
                                    add_time_ids = torch.cat(
                                        [compute_time_ids(s, r, c) for s, r, c in zip(batch["original_sizes"], batch["resized_sizes"], batch["crop_top_lefts"])]
                                    ).to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype)

                                    timesteps小 = min(range(0, 1000), key=lambda x: abs(alpha - sdxl_time_to_alpha[x]))

                                    教师pred = 教师pipeline.unet(
                                        noisy_model_input小.to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype),
                                        timesteps小,
                                        batch['sdxl_prompt_embeds'].to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype),
                                        added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": batch['sdxl_pooled_prompt_embeds'].to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype)},
                                        return_dict=False,
                                    )[0].detach().clone().to(torch.float32)
                                    if args.teacher_cfg > 1:
                                        教师pred_uncond = 教师pipeline.unet(
                                            noisy_model_input小.to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype),
                                            timesteps小,
                                            teacher_nega_prompt_embeds.to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype),
                                            added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": teacher_nega_pooled_prompt_embeds.to(device=教师pipeline.unet.device, dtype=教师pipeline.unet.dtype)},
                                            return_dict=False,
                                        )[0].detach().clone().to(torch.float32)
                                        教师pred = 教师pred_uncond + args.teacher_cfg * (教师pred - 教师pred_uncond)
                                    教师pred_x0 = (noisy_model_input小 - beta**0.5 * 教师pred) / alpha**0.5

                            latents_to_decode = 教师pred_x0 / 教师pipeline.vae.config.scaling_factor
                            image_pixels = 教师pipeline.vae.decode(latents_to_decode, return_dict=False)[0]

                            # if args.quick_test:
                            #     print('-'*10, f'对比', '-'*10)
                            #     print(f'时间 {timesteps=} {sigmas=}', )
                            #     print('noisy_model_input小', noisy_model_input小.mean(), noisy_model_input小.var())
                            #     print('教师pred', 教师pred.mean(), 教师pred.var())
                            #     print('noise小', noise小.mean(), noise小.var())
                            #     print('教师pred_x0', 教师pred_x0.mean(), 教师pred_x0.var())
                            #     image = 教师pipeline.image_processor.postprocess(image_pixels, output_type='pil')[0]
                            #     image.save(f'fk/{int(timesteps)}_教师pred_x0_pixels.png')
                            #     with open(f'fk/{int(timesteps)}_prompt.txt', 'w', encoding='utf8') as f:
                            #         f.write(str(batch['teacher_prompts']))

                            image_pixels = torch.clamp(image_pixels, min=-1.0, max=1.0)
                            target_x0_sd3 = flux_vae_encode(vae, image_pixels)

                            noisy_model_input = (1.0 - sigmas) * target_x0_sd3 + sigmas * noise
                            batch['timesteps'] = timesteps.to('cpu')
                            batch['sigmas'] = sigmas
                            batch['noisy_model_input'] = Flux2KleinPipeline._patchify_latents(noisy_model_input.to('cpu'))
                            batch['target_v'] = Flux2KleinPipeline._patchify_latents(noise - target_x0_sd3)
                        for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                            i.to('cpu')
                    with 计时(accelerator, global_step, 'clean'):
                        clean()
                transformer.to(accelerator.device)
            else:
                yield from batch_buffer
                batch_buffer = []

    train_dataloader_cycle = cycle(train_dataloader)
    train_dataloader_超 = 超源(train_dataloader_cycle, accelerator)

    transformer.train()
    while global_step <= args.max_train_steps:
        batch = next(train_dataloader_超)
        with 计时(accelerator, global_step, 'step'):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                timesteps = batch['timesteps'].to(accelerator.device)
                noisy_model_input = batch['noisy_model_input'].to(accelerator.device)
                target = batch['target_v'].to(accelerator.device)
                sigmas = batch['sigmas']
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_model_input).to(device=accelerator.device)

                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                    guidance = guidance.expand(noisy_model_input.shape[0])
                else:
                    guidance = None

                超prompt_embeds = torch.cat([batch['prompt_embeds'].to(accelerator.device), F.pad(batch['sdxl_prompt_embeds'].to(accelerator.device), (0, 7680 - 2048))], dim=1)
                n = 超prompt_embeds.shape[1]
                超text_ids = torch.zeros(1, n, 4, dtype=torch.long, device=accelerator.device)
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
                model_pred_v = model_pred_v[:, : packed_noisy_model_input.size(1) :]
                model_pred_v = Flux2KleinPipeline._unpack_latents_with_ids(model_pred_v, model_input_ids)

                # if args.quick_test:
                #     model_pred_x0 = model_pred_v * (-sigmas) + noisy_model_input
                #     with torch.no_grad():
                #         vae.to(accelerator.device)
                #         for latent, 名字 in [(model_pred_x0, '学生pred_x0'), (noisy_model_input, 'xt')]:
                #             latent = latent * latents_bn_std + latents_bn_mean
                #             image = vae.decode(Flux2KleinPipeline._unpatchify_latents(latent).to(device=vae.device, dtype=vae.dtype), return_dict=False)[0]
                #             image = text_encoding_pipeline.image_processor.postprocess(image, output_type='pil')[0]
                #             image.save(f'fk/{int(timesteps)}_{名字}.png')
                #         vae.to('cpu')

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = torch.mean(
                    (weighting.float() * (model_pred_v.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
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
                vae.to(accelerator.device)
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
                # log_validation(pipeline, accelerator, global_step, 3)
                log_validation(pipeline, accelerator, global_step, 5)

                for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                    i.to('cpu')

                transformer.to(accelerator.device)
                vae.to(accelerator.device)
                text_encoder.to('cpu')
                optimizer_to_device(optimizer, accelerator.device)
                clean()

        if accelerator.sync_gradients:
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "grad_norm": grad_norm.item()}
            progress_bar.set_postfix(**logs)
            if timesteps[0] > 800:
                logs = logs | {
                    'grad_norm_大于800': logs['grad_norm'],
                    'loss_大于800': logs['loss'],
                }
            else:
                logs = logs | {
                    'grad_norm_小于800': logs['grad_norm'],
                    'loss_小于800': logs['loss'],
                }
            accelerator.log(logs, step=global_step)
            progress_bar.update(1)
            accelerator.log({"len_tag": batch['prompts'][0].count(','), "t": timesteps[0]}, step=global_step)
            global_step += 1
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
                if platform.system() == 'Windows' and global_step % (args.checkpointing_steps * 10) == 0:
                    exit()

        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
