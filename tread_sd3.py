import torch
from diffusers.models.transformers.transformer_sd3 import JointTransformerBlock


记忆 = None

def patch_sd3_tread(model, a=4, b=18, p=0.25):
    for i, block in enumerate(model.transformer_blocks):
        block._tread_layer_index = i
    _原forward = JointTransformerBlock.forward
    def new_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        global 记忆
        current_layer_idx = self._tread_layer_index
        if torch.is_inference_mode_enabled():
            return _原forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)
        res = _原forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)
        if current_layer_idx == a:
            记忆 = res 
        elif current_layer_idx == b:
            for x1, x2 in zip(res, 记忆):
                mask = torch.rand_like(x1, dtype=torch.float) < p
                x1[mask] = x2[mask]
        return res
    JointTransformerBlock.forward = new_forward
