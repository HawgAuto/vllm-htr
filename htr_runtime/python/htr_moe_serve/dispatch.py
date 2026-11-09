"""Selected grouped-MoE dispatcher over the retained native kernels.

Unsupported shapes use the original partitioned route. The serving wrapper
provides a nonrecursive original-forward proxy and live parity checks.
"""
CANDIDATE_ROWS=frozenset((12,16,24,32,40))


def forward(R,candidate,ar,x,ids,w):
    if x.shape[0] not in CANDIDATE_ROWS or ar['_kf']!=4:
        return R.forward(ar,x,ids,w)
    if (R.BPS_GU,R.BPS_DN,R.FENCE_RA,R.FUSE,R.V2_MIN,R._htr_partition4)!=(5,1,1,True,48,True):
        raise RuntimeError('Grouped MoE requires the qualified partition4/fused native arithmetic and split contract')
    return candidate.forward(R,ar,x,ids,w)
