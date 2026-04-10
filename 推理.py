import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, Flux2KleinPipeline

from 哭 import 哭model

# 原 = "R:/models/FLUX.2-klein-base-4B"
原 = "C:/Users/Administrator/Desktop/FLUX.2-klein-base-4B"
# 新 = "R:/lora切/fk/5F3-243-AFC-muon-lr1.2e-05-0.75-6-drop0.1&0.1&0.6-bf16-SS1.0-n4-cfg1.5-logit_normal_-2.2_1.3-学人0.5-Muon40.0-哭2/checkpoint-420000/model.safetensors"
新 = "R:/lora切/fk/5F3-243-AFC-muon-lr6e-06-0.8-6-drop0.1&0.1&0.6-bf16-SS1.0-n4-cfg1.5-logit_normal_-1.9_1.3-学人0.5-Muon40.0-哭/checkpoint-608000/model.safetensors"
# 新 = "C:/Users/Administrator/Desktop/checkpoint-528000/model.safetensors"
sdxl = "C:/Users/Administrator/Desktop/models/waiIllustriousSDXL_v160.safetensors"


class 哭Pipeline(Flux2KleinPipeline):
    @torch.no_grad()
    def __call__(self, *, prompt, **kwargs):
        prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            max_sequence_length=200,
            text_encoder_out_layers=[10, 20, 30],
        )
        sdxl_prompt_embeds, *_ = 教师pipeline.encode_prompt(prompt)
        超prompt_embeds = torch.cat([prompt_embeds, F.pad(sdxl_prompt_embeds.to('cuda'), (0, 7680 - 2048))], dim=1).to(torch.bfloat16)
        return super().__call__(prompt_embeds=超prompt_embeds, **kwargs)


transformer = 哭model.from_pretrained(原, subfolder="transformer")
state_dict = load_file(新, device='cuda')
transformer.load_state_dict(state_dict, assign=True)

pipeline = 哭Pipeline.from_pretrained(原, transformer=transformer, torch_dtype=torch.bfloat16)
pipeline.to('cuda')

教师pipeline = StableDiffusionXLPipeline.from_single_file(sdxl, torch_dtype=torch.float16)


validation_prompt = [
    ('1girl, kisaki (blue archive), eating baozi, sitting, indoors', 1),
    ('1girl, momoi (blue archive), typing on keyboard, computer, animal ear headphones, sitting, angry, indoors, newest', 2),
    ('1girl, yuuka (blue archive), holding cup, sitting, indoors, kantoku, newest', 3),
    ('1girl, hoshino (blue archive), eating pizza, sitting, indoors', 4),
    ('1girl, kisaki (blue archive), 校服, 室外, 拿着饮料', 1),
]


for i, (prompt, seed) in enumerate(validation_prompt):
    pipeline(
        prompt=prompt,
        generator=torch.Generator(device='cpu').manual_seed(seed),
        num_inference_steps=20,
        guidance_scale=5,
        width=960,
        height=1024,
    ).images[0].save(f'output_{i}.png')
