import mindspore as ms
import mindspore.nn as nn

from typing import Dict

from layers import LoRALayer

def mark_only_lora_as_trainable(model: nn.Cell, bias: str = 'none') -> None:
    for n, p in model.parameters_and_names():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.parameters_and_names():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for name, cell in model.cells_and_names():
            if isinstance(cell, LoRALayer) and hasattr(cell, 'bias') and cell.bias is not None:
                cell.bias.requires_grad = True
    else:
        raise NotImplementedError

def lora_state_dict(model: nn.Cell, bias: str = 'none') -> Dict[str, ms.Tensor]:
    my_state_dict = model.parameters_dict()
    if bias == 'none':
        return {k: v for k, v in my_state_dict.items() if 'lora_' in k}
    elif bias == 'all':
        return {k: v for k, v in my_state_dict.items() if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k, v in my_state_dict.items():
            if 'lora_' in k:
                to_return[k] = v
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError