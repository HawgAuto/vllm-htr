"""Opaque packed W8A16 op. No CPU or whole-weight BF16 fallback."""
import torch
from .wire import Packet,validate


def check_metadata(x,q,s,z,out_features):
    validate(Packet(q,s,z),values=False)
    if x.dtype!=torch.bfloat16 or x.ndim<2 or x.shape[-1]!=q.shape[1]:
        raise ValueError("GPTQ8 HC expects BF16 activations [...,K]")
    if not (x.device==q.device==s.device==z.device):
        raise ValueError("all operands must share a device")
    if not q.shape[0]<=out_features<q.shape[0]+16:
        raise ValueError("only the model's <16 output alignment rows may be padded")


@torch.library.custom_op('gp8hc::linear',mutates_args=())
def linear(x:torch.Tensor,q:torch.Tensor,s:torch.Tensor,z:torch.Tensor,out_features:int)->torch.Tensor:
    check_metadata(x,q,s,z,out_features)
    if x.device.type!='cuda':raise ValueError('packed HC requires a qualified GPU; no BF16 fallback')
    from .kernels import launch
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    y=launch(flat,q,s,z,out_features)
    return y.reshape(*x.shape[:-1],out_features)


@linear.register_fake
def _fake(x,q,s,z,out_features):
    check_metadata(x,q,s,z,out_features)
    return x.new_empty((*x.shape[:-1],out_features))
