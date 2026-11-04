"""Manually authored Triton kernel; identical butterfly order to donor FP32."""
import triton
import triton.language as tl

@triton.jit
def _rht256(X,S,Y,S0:tl.constexpr,S1:tl.constexpr,O0:tl.constexpr,O1:tl.constexpr,INVERSE:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1);c=tl.arange(0,256)
    v=tl.load(X+t*S0+h*S1+c).to(tl.float32)
    signs=tl.load(S+c).to(tl.float32)
    if not INVERSE:v=v*signs
    for stage in tl.static_range(8):
        step=1<<stage
        peer=tl.gather(v,c^step,axis=0)
        v=tl.where((c&step)==0,v+peer,peer-v)
    if INVERSE:v=v*signs
    tl.store(Y+t*O0+h*O1+c,v)
