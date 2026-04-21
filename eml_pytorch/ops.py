import torch
from torch.library import custom_op, register_autograd, register_fake

@custom_op("eml_pytorch::eml", mutates_args=())
def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_safe = torch.clamp(y, min=1e-8)
    return torch.exp(x) - torch.log(y_safe)

@register_fake("eml_pytorch::eml")
def _(x, y):
    return torch.empty_like(x)

def eml_setup_context(ctx, inputs, output):
    x, y = inputs
    ctx.save_for_backward(x, y)

def eml_backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    y_safe = torch.clamp(y, min=1e-8)
    grad_x = grad_output * torch.exp(x)
    grad_y = grad_output * (-1.0 / y_safe)
    return grad_x, grad_y

register_autograd("eml_pytorch::eml", eml_backward, setup_context=eml_setup_context)