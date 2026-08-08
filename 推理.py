import os
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, Flux2KleinPipeline

from 哭 import 哭model

原 = "R:/models/FLUX.2-klein-base-4B"
新 = "R:/RUM-FLUX.2-klein-4B/RUM-FLUX.2-klein-4B.safetensors"

sdxl = "C:/Users/Administrator/Desktop/models/waiNSFWIllustrious_v140.safetensors"


os.makedirs('测试输出', exist_ok=True)


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
    def __call__(self, *, prompt, negative_prompt='', **kwargs):
        return super().__call__(prompt_embeds=self.上床(prompt), negative_prompt_embeds=self.上床(negative_prompt), **kwargs)


transformer = 哭model.from_pretrained(原, subfolder="transformer")
state_dict = load_file(新, device='cuda')
transformer.load_state_dict(state_dict, assign=True)

pipeline = 哭Pipeline.from_pretrained(原, transformer=transformer, torch_dtype=torch.bfloat16)
pipeline.to('cuda')

教师pipeline = StableDiffusionXLPipeline.from_single_file(sdxl, torch_dtype=torch.float16)
教师pipeline.text_encoder.to('cuda')
教师pipeline.text_encoder_2.to('cuda')

validation_prompt = [
    ('1girl, kisaki (blue archive), holding baozi, eating, table, indoors, looking down, momoko (momopoco), liduke', 1),
    ('1girl, momoi (blue archive), typing on keyboard, computer, blue necktie, pink shoulder white sleeve, white coat, white shirt, v-shaped eyebrows, sitting on gaming chair, indoors, baram, starshadowmagician', 2),
    ('1girl, yuuka (blue archive), holding cup, black jacket, suit, blue necktie, hand twirling hair, indoors, fuzichoco, mika pikazo', 3),
    ('1girl, azusa (blue archive), holding ice cream, eating, outdoors, shopping street, black sailor collar, white shirt, light smile, huwari (dnwls3010)', 4),
]

for width in [960]:
    for height in [1152]:
        for guidance_scale in [5]:
            for num_inference_steps in [30]:
                for i, (prompt, seed) in enumerate(validation_prompt):
                    pipeline(
                        prompt=prompt,
                        generator=torch.Generator(device='cpu').manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    ).images[0].save(f'测试输出/output_{i}_cfg{guidance_scale}_seed{seed}_n{num_inference_steps}_{width}×{height}.png')


edit_prompt = [
    ('change style to fuzichoco', 'change background', 5),
    ('change to 1girl, hakurei reimu, black hair, ascot, remove twintails', '', 5),
    ('white hair', 'short hair', 1),
    ('change to long dress, wedding dress, holding own dress, strapless', '', 5),
    ('change to long hair', '', 2),
    ('beach, outdoors', '', 5),
]

for num_inference_steps in [10]:
    for i, (prompt, negative_prompt, guidance_scale) in enumerate(edit_prompt):
        pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.open("./img/靓仔.png"),
            generator=torch.Generator(device='cpu').manual_seed(1),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0].save(f'测试输出/编辑_output_{i}_cfg{guidance_scale}_n{num_inference_steps}.png')
