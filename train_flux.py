import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

from datasets import load_dataset
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import upload_folder
from huggingface_hub.utils import insecure_hashlib
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel, StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from compel import Compel, ReturnedEmbeddingsType

from dan后处理 import dan后处理
from common import 计时, 哈, cycle, clean, sdxl_time_to_alpha, 生成optimizer, 评测pipeline, add_image_jpeg, validation_prompt, clone_state_dict, compute_time_ids
from common import encode_prompt as encode_prompt_sdxl

import muon
muon.zeropower_via_newtonschulz5 = torch.compile(muon.zeropower_via_newtonschulz5)


logger = get_logger(__name__)


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant)
    text_encoder_two = class_two.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant)
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    accelerator,
    global_step,
    guidance_scale=5,
):
    images = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
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
                分数 = 评测pipeline(pipeline, n_iter=args.validation_n_iter, guidance_scale_range=(guidance_scale, guidance_scale+2))
                accelerator.log({f"分数-cfg{guidance_scale}": 分数}, step=global_step)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            concat_image = np.concatenate([np.asarray(img) for img in images], axis=1)
            add_image_jpeg(tracker.writer, f"validation-cfg{guidance_scale}", concat_image, global_step)
    del pipeline
    clean()
    return images


def 生成dataset(accelerator, drop_tag_rate, drop_char_feature_rate) -> tuple:
    d后 = dan后处理(drop_tag_rate, drop_char_feature_rate, (576, 1344))
    dataset = load_dataset(
        "imagefolder",
        data_files={"train": os.path.join(args.train_data_dir, "**")},
    )
    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(d后.preprocess_train, output_all_columns=True)
    return train_dataset


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
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
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=200,
        help="Maximum sequence length to use with with the T5 text encoder",
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
        "--prefetch_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
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
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
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
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
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
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
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
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
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
        "--use_teacher_text_encoder",
        action="store_true",
        default=False,
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
    original_sizes = [example["original_sizes"] for example in examples]
    resized_sizes = [example["resized_sizes"] for example in examples]
    crop_top_lefts = [example["crop_top_lefts"] for example in examples]
    原本images = [example["原本images"] for example in examples]
    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "resized_sizes": resized_sizes,
        "crop_top_lefts": crop_top_lefts,
        "原本images": 原本images,
    }
    return batch


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    if args.quick_test:
        global validation_prompt
        validation_prompt = validation_prompt[:1]
        args.validation_steps = 5
        args.prefetch_steps = 2
        args.seed = 1
        args.validation_n_iter = 2
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='tensorboard',
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant)

    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # weight_dtype = torch.float32
    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    transformer.to('cpu', dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder_one.to('cpu', dtype=torch.float32)
    text_encoder_one.to('cpu', dtype=weight_dtype)
    text_encoder_two.to('cpu', dtype=weight_dtype)

    教师pipeline = StableDiffusionXLPipeline.from_single_file(args.teacher_model_name_or_path, torch_dtype=torch.float16)
    教师pipeline的compel = Compel(truncate_long_prompts=False, tokenizer=[教师pipeline.tokenizer, 教师pipeline.tokenizer_2], text_encoder=[教师pipeline.text_encoder, 教师pipeline.text_encoder_2],  returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    教师pipeline.vae.to(dtype=torch.float32)
    教师pipeline.unet.requires_grad_(False)
    教师pipeline.vae.requires_grad_(False)
    教师pipeline.text_encoder.requires_grad_(False)
    教师pipeline.text_encoder_2.requires_grad_(False)

    if args.use_teacher_text_encoder:
        text_encoder_one.load_state_dict(clone_state_dict(教师pipeline.text_encoder.state_dict()), strict=True)

    for k in ['transformer', 'vae', 'text_encoder_one', 'text_encoder_two', '教师pipeline.unet', '教师pipeline.vae', '教师pipeline.text_encoder', '教师pipeline.text_encoder_2']:
        v = eval(k)
        print(f'{k}({type(v).__name__})参数量: {v.num_parameters(only_trainable=False) / 1e9:.2f} B，dtype={v.dtype}')

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()

            if isinstance(unwrap_model(model), FluxTransformer2DModel):
                load_model = FluxTransformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = 生成optimizer(args.optimizer, transformer, args.adam_beta1, args.adam_beta2, args.adam_weight_decay, args.adam_epsilon, args.learning_rate, args.learning_rate * 40)
    train_dataset = 生成dataset(accelerator, args.drop_tag_rate, args.drop_char_feature_rate)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    特征 = f'{哈(args.train_data_dir)}-{哈(args.pretrained_model_name_or_path)}-{哈(args.teacher_model_name_or_path)}-{args.optimizer}-lr{args.learning_rate}-{args.lr_num_cycles}-drop{args.drop_text_rate}&{args.drop_tag_rate}&{args.drop_char_feature_rate}-{args.mixed_precision}-{args.lr_scheduler}-SS{args.sigmas_scale}-n{args.inference_steps}' + '-TEST'*bool(args.quick_test) + '-te'*bool(args.use_teacher_text_encoder) 
    if accelerator.is_main_process:
        accelerator.init_trackers(f'flux-{特征}', config=vars(args))
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
    global_step = 0

    if args.resume_from_checkpoint == 'latest':
        if 候选checkpoint := [*Path(checkpoint_dir).glob('checkpoint-*')]:
            args.resume_from_checkpoint = str(max(候选checkpoint, key=lambda i:int(i.stem.split('-')[-1])))
        else:
            args.resume_from_checkpoint = None
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("-")[-1])
        initial_global_step = global_step
    else:
        initial_global_step = 0

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

    def 超源(it, text_encoders, tokenizers, accelerator):
        batch_buffer = []
        while True:
            if not batch_buffer:
                transformer.to('cpu')
                with torch.no_grad():
                    for _ in range(args.prefetch_steps):
                        batch = next(it)
                        batch_buffer.append(batch)
                        if random.random() < args.drop_text_rate:
                            batch['prompts'] = ['' for _ in batch['prompts']]

                    for i in text_encoders:
                        i.to(accelerator.device)
                    for batch in batch_buffer:
                        batch['prompt_embeds'], batch['pooled_prompt_embeds'], batch['text_ids'] = compute_text_embeddings(batch['prompts'], text_encoders, tokenizers)
                    for i in text_encoders:
                        i.to('cpu')

                    for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                        i.to(accelerator.device)
                    for batch in batch_buffer:
                        batch['sdxl_prompt_embeds'], batch['sdxl_pooled_prompt_embeds'] = [i.cpu() for i in encode_prompt_sdxl(batch['prompts'], 教师pipeline的compel)]
                    for i in [教师pipeline.text_encoder, 教师pipeline.text_encoder_2]:
                        i.to('cpu')
                    clean()

                    for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                        i.to(accelerator.device)
                    for batch in batch_buffer:
                        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                        with 计时(accelerator, global_step, 'vae_time'):
                            x0 = vae.encode(pixel_values).latent_dist.mean
                            x0 = (x0 - vae.config.shift_factor) * vae.config.scaling_factor
                            x0 = x0.to(torch.float32)
                        with 计时(accelerator, global_step, 'vae2_time'):
                            x0小 = 教师pipeline.vae.encode(pixel_values.to(教师pipeline.vae.dtype)).latent_dist.mean
                            x0小 = x0小 * 教师pipeline.vae.config.scaling_factor
                            x0小 = x0小.to(torch.float32)
                        noise = torch.randn_like(x0)
                        noise小 = noise.unflatten(1, (4, 4)).mean(dim=2) * 2
                        bsz = x0.shape[0]
                        assert bsz == 1

                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=args.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                            mode_scale=args.mode_scale,
                        )
                        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                        timesteps = noise_scheduler_copy.timesteps[indices].to(device=x0.device)

                        sigmas = get_sigmas(timesteps, n_dim=x0.ndim, dtype=x0.dtype)
                        noisy_model_input = (1.0 - sigmas) * x0 + sigmas * noise

                        x0小_std = x0小.std()
                        x0_std = x0.std()
                        sigmas小 = sigmas*x0小_std / (-sigmas*x0_std + sigmas*x0小_std + x0_std) * args.sigmas_scale
                        估计方差 = (1.0 - sigmas小) ** 2 + sigmas小 ** 2
                        k = 1 / 估计方差**0.5

                        初始beta = (sigmas小 * k) ** 2
                        教师pred_x0 = x0小
                        torch.cuda.empty_cache()
                        for i in range(args.inference_steps):
                            beta = 初始beta * (1 - i / args.inference_steps)
                            alpha = 1 - beta
                            if i == 0:
                                noisy_model_input小 = alpha**0.5 * 教师pred_x0 + beta**0.5 * noise小
                            else:
                                noisy_model_input小 = alpha**0.5 * 教师pred_x0 + beta**0.5 * 教师pred
                            with 计时(accelerator, global_step, 'unet_time'):
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

                                教师pred_x0 = (noisy_model_input小 - beta**0.5 * 教师pred) / alpha**0.5

                        latents_to_decode = 教师pred_x0 / 教师pipeline.vae.config.scaling_factor
                        image_pixels = 教师pipeline.vae.decode(latents_to_decode, return_dict=False)[0]

                        if args.quick_test:
                            print('-'*10, f'对比', '-'*10)
                            print(f'时间 {timesteps=} {timesteps小=}', )
                            print(f'sigmas {sigmas=} {sigmas小=}')
                            print('noisy_model_input小', noisy_model_input小.mean(), noisy_model_input小.var())
                            print('教师pred', 教师pred.mean(), 教师pred.var())
                            print('noise小', noise小.mean(), noise小.var())
                            print('教师pred_x0', 教师pred_x0.mean(), 教师pred_x0.var())
                            print('x0小', x0小.mean(), x0小.var())
                            image = 教师pipeline.image_processor.postprocess(image_pixels, output_type='pil')[0]
                            image.save(f'rum_flux/{int(timesteps)}_教师pred_x0_pixels.png')
                            batch["原本images"][0].save(f'rum_flux/{int(timesteps)}_原本.png')

                        image_pixels = torch.clamp(image_pixels, min=-1.0, max=1.0)
                        target_dist = vae.encode(image_pixels.to(vae.dtype)).latent_dist
                        target_x0_flux = target_dist.mean
                        target_x0_flux = (target_x0_flux - vae.config.shift_factor) * vae.config.scaling_factor
                        target = target_x0_flux.to(dtype=transformer.dtype)
                        batch['timesteps'] = timesteps.to('cpu')
                        batch['noise'] = noise.to('cpu')
                        batch['noisy_model_input'] = noisy_model_input.to('cpu')
                        batch['target'] = target.to('cpu')
                        batch['sigmas'] = sigmas
                    for i in [vae, 教师pipeline.vae, 教师pipeline.unet]:
                        i.to('cpu')
                    clean()
                transformer.to(accelerator.device)
            else:
                yield from batch_buffer
                batch_buffer = []

    train_dataloader_超 = 超源(cycle(train_dataloader), text_encoders, tokenizers, accelerator)

    transformer.train()
    while global_step <= args.max_train_steps:
        with 计时(accelerator, global_step, 'step_time'):
            batch = next(train_dataloader_超)
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                model_input = batch['target'].to(accelerator.device)

                vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                packed_noisy_model_input = FluxPipeline._pack_latents(
                    model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=batch['timesteps'].to(accelerator.device) / 1000,
                    guidance=guidance,
                    pooled_projections=batch['pooled_prompt_embeds'].to(accelerator.device),
                    encoder_hidden_states=batch['prompt_embeds'].to(accelerator.device),
                    txt_ids=batch['text_ids'],
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=batch['sigmas'])

                target = batch['noise'].to(accelerator.device) - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                with 计时(accelerator, global_step, 'optimizer_step_time'):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if global_step % 40 == 20:
                clean()
                accelerator.log({"memory_allocated": torch.cuda.memory_allocated() / 1024**3, "memory_reserved": torch.cuda.memory_reserved() / 1024**3}, step=global_step)

        if accelerator.is_main_process and (global_step % args.validation_steps == 0 or global_step in [args.validation_steps // 2]):
            text_encoder_one.to(accelerator.device)
            text_encoder_two.to(accelerator.device)
            transformer.to(accelerator.device)
            vae.to(accelerator.device)
            pipeline = FluxPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=unwrap_model(text_encoder_one),
                text_encoder_2=unwrap_model(text_encoder_two),
                transformer=unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            log_validation(pipeline, accelerator, global_step)
            text_encoder_one.to('cpu')
            text_encoder_two.to('cpu')
            transformer.to(accelerator.device)
            vae.to('cpu')

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        if accelerator.sync_gradients:
            progress_bar.update(1)
            accelerator.log({"len_tag": batch['prompts'][0].count(','), "t": batch['timesteps'][0]}, step=global_step)
            global_step += 1
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
