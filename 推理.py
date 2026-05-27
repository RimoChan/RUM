import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, Flux2KleinPipeline

from 哭 import 哭model

# 原 = "R:/models/FLUX.2-klein-base-4B"
原 = "C:/Users/Administrator/Desktop/FLUX.2-klein-base-4B"
新 = r"R:\RUM-FLUX.2-klein-4B-preview\model-checkpoint-908000.safetensors"

sdxl = "C:/Users/Administrator/Desktop/models/waiNSFWIllustrious_v140.safetensors"

from diffusers.pipelines.flux2.pipeline_flux2_klein import *


class 哭Pipeline(Flux2KleinPipeline):
    def 上床(self, prompt):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=200,
            text_encoder_out_layers=[10, 20, 30],
        )
        sdxl_prompt_embeds, *_ = 教师pipeline.encode_prompt(prompt)
        return torch.cat([prompt_embeds, F.pad(sdxl_prompt_embeds.to('cuda'), (0, 7680 - 2048))], dim=1).to(torch.bfloat16)

    @torch.inference_mode()
    def __call__(self, *, prompt, **kwargs):
        return super().__call__(prompt_embeds=self.上床(prompt), negative_prompt_embeds=self.上床(''), **kwargs)


transformer = 哭model.from_pretrained(原, subfolder="transformer")
state_dict = load_file(新, device='cuda')
transformer.load_state_dict(state_dict, assign=True)

pipeline = 哭Pipeline.from_pretrained(原, transformer=transformer, torch_dtype=torch.bfloat16)
pipeline.to('cuda')

教师pipeline = StableDiffusionXLPipeline.from_single_file(sdxl, torch_dtype=torch.float16)


validation_prompt = [
    ('1girl, kisaki (blue archive), holding baozi, eating, sitting, indoors, momoko (momopoco)', 1),
    ('1girl, momoi (blue archive), typing on keyboard, computer, animal ear headphones, sitting, angry, indoors, mika pikazo', 2),
    ('1girl, yuuka (blue archive), holding cup, sitting, indoors, fuzichoco', 3),
    ('1girl, mika (blue archive), holding pizza, eating, sitting, indoors, huwari (dnwls3010)', 4),
]


for width in [960]:
    for height in [1024]:
        for guidance_scale in [9]:
            for num_inference_steps in [20]:
                for i, (prompt, seed) in enumerate(validation_prompt):
                    pipeline(
                        prompt=prompt,
                        generator=torch.Generator(device='cpu').manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    ).images[0].save(f'测试输出/output_{i}_cfg{guidance_scale}_seed{seed}_n{num_inference_steps}_{width}×{height}.png')
