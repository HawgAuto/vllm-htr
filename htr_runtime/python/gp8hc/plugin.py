"""Opt-in vLLM HC adapter, leaving the HTR/Quark config identities intact."""
import torch
from torch.nn import Parameter
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase,LinearMethodBase,UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs
from .wire import WIRE,GROUP,Packet,validate

SELECTED=[]


def current_overlay():
    from vllm.config import get_current_vllm_config_or_none
    vc=get_current_vllm_config_or_none()
    qc=getattr(vc.model_config.hf_config,'quantization_config',None) if vc is not None else None
    return qc.get('gptq8_hc_overlay') if isinstance(qc,dict) else None


class GPTQ8HCMethod(LinearMethodBase):
    def __init__(self,spec):self.spec=spec

    def create_weights(self,layer,input_size_per_partition,output_partition_sizes,input_size,output_size,params_dtype,**extra_weight_attrs):
        n=input_size_per_partition;out=sum(output_partition_sizes);m=self.spec['m']
        if n!=input_size or out!=output_size:raise ValueError('GPTQ8 HC supports TP1 only')
        if n!=self.spec['n'] or not m<=out<m+16 or params_dtype!=torch.bfloat16:
            raise ValueError('invalid HC input/output/dtype contract')
        shapes={'gp8_qweight':((m,n),torch.uint8),'gp8_scales':((m,(n+GROUP-1)//GROUP),torch.float16),'gp8_zeros':((m,(n+GROUP-1)//GROUP),torch.uint8)}
        layer._gp8_loaded=set()
        for nm,(shape,dtype) in shapes.items():
            param=Parameter(torch.empty(shape,dtype=dtype),requires_grad=False)
            def load(p,data,shard_id=None,nm=nm):
                if shard_id is not None or getattr(data,'shard_id',None) is not None:
                    raise ValueError('HC packets are already fused; shards forbidden')
                if data.shape!=p.shape or data.dtype!=p.dtype:raise ValueError('HC payload shape/dtype mismatch')
                p.data.copy_(data);layer._gp8_loaded.add(nm)
            layer.register_parameter(nm,param);set_weight_attrs(param,{'weight_loader':load})
        layer._gp8_out=out
        layer.register_buffer('weight',torch.zeros(1,dtype=torch.uint8).expand(out,n),persistent=False)
        layer._gp8_weight_carrier=True
        layer._htr_weight_carrier=True  # compatibility with existing census; not HTR weight storage

    def process_weights_after_loading(self,layer):
        if layer._gp8_loaded!={'gp8_qweight','gp8_scales','gp8_zeros'}:raise ValueError('incomplete HC packet load')
        validate(Packet(layer.gp8_qweight,layer.gp8_scales,layer.gp8_zeros))
        from . import hipdec
        if hipdec.ENABLED or hipdec.PENABLED:  # E22 (+K1d prefill): packed (scale|zero<<16) u32 per (row, group) for the HIP decode kernels
            layer.register_buffer('gp8_sz',hipdec.pack_sz(layer.gp8_scales,layer.gp8_zeros),persistent=False)

    def apply(self,layer,x,bias=None):
        if bias is not None:raise ValueError('HC recipe has no bias')
        from .ops import linear
        return linear(x,layer.gp8_qweight,layer.gp8_scales,layer.gp8_zeros,layer._gp8_out)


class GPTQ8HCConfig(QuantizationConfig):
    def __init__(self,mods):super().__init__();self.mods=mods
    def get_name(self):return 'gp8hc'
    def get_supported_act_dtypes(self):return [torch.bfloat16]
    @classmethod
    def get_min_capability(cls):return 0
    @staticmethod
    def get_config_filenames():return []
    @classmethod
    def from_config(cls,config):raise NotImplementedError('constructor-scoped only')
    def get_quant_method(self,layer,prefix):
        if isinstance(layer,LinearBase) and prefix in self.mods:
            SELECTED.append(prefix)
            return GPTQ8HCMethod(self.mods[prefix])
        return UnquantizedLinearMethod() if isinstance(layer,LinearBase) else None


def register():
    import htr.vllm_plugin as hp  # ensure incumbent constructor wrappers are installed first
    from vllm.models.qwen4_exp.amd import hyperconnection as hcm
    GR=hcm.GatedResidual
    if getattr(GR.__init__,'_gp8hc',False):return
    orig=GR.__init__
    def init(self,*args,**kw):
        prefix=kw.get('prefix',args[2] if len(args)>2 else '')
        ov=current_overlay()
        if not ov or hp._DRAFT_BUILD[0] or 'mtp' in prefix.split('.'):
            return orig(self,*args,**kw)
        if ov.get('wire')!=WIRE or ov.get('group_size')!=GROUP:raise ValueError('unknown GPTQ8 HC wire')
        mods=ov.get('modules',{})
        if not any(k.startswith(prefix+'.') for k in mods):return orig(self,*args,**kw)
        if not prefix.startswith('language_model.model.'):raise ValueError('invalid target HC prefix')
        htrhc=(hp._current_htr_overlay() or {}).get('hc',{}).get('modules',{})
        if set(htrhc)&set(mods):raise ValueError('overlapping HTR/GPTQ8 HC recipes')
        helper=hcm._w8;old=helper.config_or_none
        if old() is not None:raise ValueError('GPTQ8 HC conflicts with DIRECT_W8A16')
        helper.config_or_none=lambda:GPTQ8HCConfig(mods)
        try:return orig(self,*args,**kw)
        finally:helper.config_or_none=old
    init._gp8hc=True;init._gp8hc_orig=orig
    GR.__init__=init
    from . import hipdec
    hipdec.install()
