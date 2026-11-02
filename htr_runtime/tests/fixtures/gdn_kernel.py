import triton
import triton.language as tl

@triton.jit
def triton_per_fused__to_copy_add_linear_mean_mul_pow_rsqrt_sigmoid_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (((r0_1 + 128*x0) % (6144*ks0))), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (((((r0_1 + 128*x0) % 6144)) % 128)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (16384*(x0 // 48) + (((r0_1 + 128*x0) % 6144))), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1, 1], 128.0, tl.float32)
    tmp10 = (tmp6 / tmp9)
    tmp11 = tl.full([1, 1], 1e-06, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tl.rsqrt(tmp12)
    tmp14 = tmp8 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 * tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.sigmoid(tmp19)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr1 + (((r0_1 + 128*x0) % (6144*ks0))), tmp22, xmask)
