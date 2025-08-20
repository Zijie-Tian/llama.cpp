import numpy as np
from typing import List, Optional, Tuple


def preprocess_weights(
    w: np.ndarray,
    scales: np.ndarray,
    zeros: Optional[np.ndarray] = None,
    bits: int = 4,
    g: int = 4,
    bm: int = 512,
    kfactor: int = 16,
    simd_n_in: int = 16,
    simd_n_out: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offline preprocess the weights before inference.

    Parameters
    ----------
    w : np.ndarray
        Quantized weights of shape (M, K) and type "uint8".
        Add a bias of 2^(bits-1) to the original int1/2/3/4 values to convert it to uint values.
        E.g., add a bias of 2 to int2: -2, -1, 0, 1 -> 0, 1, 2, 3
    scales: np.ndarray
        Quantization scales of shape (M, K // group_size) or (m_groups,) and type float32/16.
    zeros: np.ndarray
        Same shape and type with scales.
        If None, the actual zero points will be 2^(bits-1) * scales;
        if not None, the actual zero points will be zeros + 2^(bits-1) * scales.
        E.g., before passing the zeros from BitDistiller/GPTQ, you need to modify it as following:
        `zeros = (zeros - (2 ** (bits - 1))) * scales`
    bits: int
        Number of bits for each quantized element
    g: int
        Group size of LUT
    bm: int
        Tuned tiling size of M
    kfactor: int
        Tuned tiling size of K
    simd_width: int
        128 for ARM NEON

    Returns
    -------
    w: np.ndarray
        Permuted weights
    scales: np.ndarray
        Permuted scales
    """
    assert w.dtype == "uint8"

    M, K = w.shape
    M = M * bits
    ngroups_per_elem = 8 // g

    #! (M // bits, K, bits)
    w = np.stack([(w >> ib) & 1 for ib in range(bits)], axis=-1)
    # (M // bits, K, bits) -> (M // bits, bits, K) -> (M // bits, bits, K // g, g)
    w = w.transpose(0, 2, 1).reshape(M // bits, bits, K // g, g)
    w = sum([(w[:, :, :, ig] << ig) for ig in range(g)
             ])  #! After this, each element will containes one group.

    # #> Test code
    # num_elem_w = len(w.flatten())
    # w = np.arange(num_elem_w).reshape(w.shape)

    # 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
    # for bits=3
    # bit0: [0, 8), bit1: [8, 16), bit2: [16, 24), bit0: [24, 32)
    # (M // bits // simd_n_float16, bits, simd_n_float16, K // g)
    w = w.reshape(M // bits // simd_n_out, simd_n_out, bits,
                  K // g).transpose(0, 2, 1, 3)
    mgroup = ngroups_per_elem * simd_n_in
    #! After this, w[:, b, :, :] will contains specific bit slice.
    w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in,
                  K // g).transpose(0, 2, 1, 3)

    # import pdb; pdb.set_trace()
    #             0        1             2          3                 4                  5
    w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem,
                  K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
    # w shape = (M // bm, K // g // kfactor, bm // mgroup, kfactor, simd_n_in, ngroups_per_elem)
    w = sum([(w[:, :, :, :, :, ng] << (ng * g))
             for ng in range(ngroups_per_elem)])
    w = w.reshape(M // bm, K // g // kfactor, bm // mgroup, kfactor,
                  simd_n_in)  # > Put simd_n_in into last dim.
    # input size of current TVM API
    w = w.reshape(M // bm, K // g, bm // ngroups_per_elem)

    # Check if scales shape is [M//bits, K] (per-channel)
    if scales.shape[1] == K:
        # Per-channel scales case [M//bits, K]
        scales = scales.reshape(M // bm, bm // bits, K).transpose(0, 2, 1)
        scales = scales.reshape(M // bm, K, bm // bits // simd_n_out,
                                simd_n_out)

        if zeros is not None:
            # zeros also has shape [M//bits, K]
            zeros = zeros.reshape(M // bm, bm // bits, K).transpose(0, 2, 1)
            zeros = zeros.reshape(M // bm, K, bm // bits // simd_n_out,
                                  simd_n_out)
            scales = np.stack([scales, zeros], axis=-2)
        # Final reshape
        scales = scales.reshape(M // bm, K, -1)
    elif scales.size >= M // bits:
        # Original per-group case [M//bits, K//group_size]
        group_size = K // scales.shape[1]
        scales = scales.reshape(M // bm, bm // bits,
                                K // group_size).transpose(0, 2, 1)
        scales = scales.reshape(M // bm, K // group_size,
                                bm // bits // simd_n_out, simd_n_out)
        if zeros is not None:
            zeros = zeros.reshape(M // bm, bm // bits,
                                  K // group_size).transpose(0, 2, 1)
            zeros = zeros.reshape(M // bm, K // group_size,
                                  bm // bits // simd_n_out, simd_n_out)
            scales = np.stack([scales, zeros], axis=-2)
        # input size of current TVM API
        scales = scales.reshape(M // bm, K // group_size, -1)
    else:
        if zeros is not None:
            scales = np.concatenate([scales, zeros])
    return w, scales


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]


def nmse(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return np.mean(np.square(a - b)) / np.mean(np.square(a))


def compute_error(x, y):
    Ps = np.linalg.norm(x)
    Pn = np.linalg.norm(x - y)
    return 20 * np.log10(Ps / Pn)


def print_binary(array):
    # np.set_printoptions(threshold=np.inf)  # 不省略任何元素
    binary_array = np.vectorize(lambda x: format(x if x >= 0 else
                                                 (1 << 8) + x, "08b"))(array)
    print(binary_array)


np.random.seed(21)
np.set_printoptions(precision=2, suppress=True)

# # 参数设置
# bits = 2
# M = 4096 * bits
# N = 1
# K = 4096               #> K >= g * kfactor && K >= act_group_size && K >= group_size
# g = 4
# bm = 256
# simd_n_in = 16
# simd_n_out = 8
# kfactor = 16
# act_group_size = 64  #> act_group_size >= g
# group_size = 128
# out_dtype = 'float16'
# dtype = 'int8'
# zero_point = False
# m_groups = -1

# # 参数设置
# bits = 4
# M = 128 * bits
# N = 1
# K = 128               #> K >= g * kfactor && K >= act_group_size && K >= group_size
# g = 4
# bm = 128
# simd_n_in = 16
# simd_n_out = 8
# kfactor = 16
# act_group_size = 128  #> act_group_size >= g
# group_size = 128
# out_dtype = 'float16'
# dtype = 'int8'
# zero_point = False
# m_groups = -1

bits = 2
M = 256 * bits
N = 1
K = 128  # > K >= g * kfactor && K >= act_group_size && K >= group_size
g = 4
bm = 128 * bits
simd_n_in = 16
simd_n_out = 8
kfactor = 16
act_group_size = 4  # > act_group_size >= g
group_size = 4  # Changed from 128 to 4 for multi-channel quantization
out_dtype = "float16"
dtype = "int8"
zero_point = True
m_groups = -1


def quantize_weight_per_tensor(weight_fp16, bits=2):
    """
    q_max = 2 ** (bits - 1) - 1

    scale = max(abs(weight)) / q_max
    weight = round(weight / scale) * scale

    weight_int8 in [-q_max, q_max]
    """
    scales = np.max(np.abs(weight_fp16))
    q_max = 2**(bits - 1) - 1
    scales = max(scales, 1e-5) / q_max
    weight_int8 = np.round(weight_fp16 / scales).clip(-q_max,
                                                      q_max).astype(np.int8)
    return weight_int8, scales


def dequantize_weight_per_tensor(weight_int8, scales):
    return weight_int8.astype(np.float16) * scales


def quantize_weight_per_group(weight_fp16, bits=2):
    """
    q_max = 2 ** (bits - 1) - 1

    scale = max(abs(weight)) / q_max
    weight = round(weight / scale) * scale

    weight_int8 in [-q_max, q_max]
    """
    q_max = 2**(
        bits - 1
    ) - 1  # NOTE: We limit in range [ - 2 ** (bits - 1), 2 ** (bits - 1) - 1 ]

    gmin, gmax = weight_fp16.min(axis=-1), weight_fp16.max(axis=-1)
    scales = np.expand_dims(np.clip((gmax - gmin) / (q_max * 2), 1e-5, None),
                            axis=-1)
    zero_point = np.expand_dims(gmin, axis=-1)
    weight_int8 = np.round(
        (weight_fp16 - zero_point) / scales).clip(-q_max,
                                                  q_max).astype(np.int8)

    return weight_int8, scales, zero_point


def dequantize_weight_per_group(weight_int8, scales, zero_point):
    return weight_int8.astype(np.float16) * scales + zero_point


def quantize_weight_multi_channel(weight_fp16, bits=2, group_size=4):
    """
    Multi-channel quantization: all M rows share the same scales and zero_points

    Parameters:
    -----------
    weight_fp16: np.ndarray of shape (M, K)
    bits: int, number of bits for quantization
    group_size: int, size of each group along K dimension

    Returns:
    --------
    weight_int8: quantized weights of shape (M, K)
    scales: scales of shape (1, K//group_size)
    zero_points: zero_points of shape (1, K//group_size)
    """
    M, K = weight_fp16.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    q_max = 2**(bits - 1) - 1

    # Reshape to (M, K//group_size, group_size)
    weight_reshaped = weight_fp16.reshape(M, K // group_size, group_size)

    # Compute min/max across all M rows and within each group
    # Shape: (K//group_size,)
    gmin = weight_reshaped.min(axis=(0, 2))
    gmax = weight_reshaped.max(axis=(0, 2))

    # Compute scales and zero_points, shape: (1, K//group_size)
    scales = np.expand_dims(np.clip((gmax - gmin) / (q_max * 2), 1e-5, None),
                            axis=0)
    zero_points = np.expand_dims(gmin, axis=0)

    # Expand for quantization
    scales_expanded = np.repeat(scales, group_size, axis=1)  # (1, K)
    zero_points_expanded = np.repeat(zero_points, group_size, axis=1)  # (1, K)

    # Quantize
    weight_int8 = np.round(
        (weight_fp16 - zero_points_expanded) / scales_expanded).clip(
            -q_max, q_max).astype(np.int8)

    return weight_int8, scales, zero_points


def dequantize_weight_multi_channel(weight_int8,
                                    scales,
                                    zero_points,
                                    group_size=4):
    """
    Dequantize multi-channel quantized weights
    """
    # Expand scales and zero_points
    scales_expanded = np.repeat(scales, group_size, axis=1)
    zero_points_expanded = np.repeat(zero_points, group_size, axis=1)

    return weight_int8.astype(
        np.float16) * scales_expanded + zero_points_expanded


weight = np.random.randn(M // bits, K).astype(out_dtype)  # FP16
# weight = np.ones((M // bits, K)).astype(out_dtype)
activation = np.random.randn(N, K).astype(out_dtype)  # FP16

# Use multi-channel quantization instead of per-group
weight_quant_mc, scales_mc, zero_point_mc = quantize_weight_multi_channel(
    weight, bits=bits, group_size=group_size)

dequantized_weight_mc = dequantize_weight_multi_channel(weight_quant_mc,
                                                        scales_mc,
                                                        zero_point_mc,
                                                        group_size=group_size)

print(f"Multi-channel quantization:")
print(f"  Scales shape: {scales_mc.shape}")  # Should be (1, K//group_size)
print(f"  Zero points shape: {zero_point_mc.shape}"
      )  # Should be (1, K//group_size)

qweight = weight_quant_mc
scales = scales_mc
zero_points = zero_point_mc

# weight_quant, scales = quantize_weight_per_tensor(weight, bits=bits)
# print("weight_quant:", weight_quant)
# print("scales:", scales)
#
# dequantized_weight = dequantize_weight_per_tensor(weight_quant, scales)
# print("dequantized_weight:", dequantized_weight)

# qweight = weight_quant
# scale = scales * np.ones((M // bits, K // group_size))

Bref = activation

# weight = np.load("weight.npy").astype(out_dtype)
# activation = np.load("activation.npy").astype(out_dtype) # FP16

# TODO : Implement real group quantization.
# qweight, scale = weight_quant_numpy(weight, -1)
# scale = scale * np.ones((M // bits, K // group_size), dtype=out_dtype)

Aref = np.round(qweight + 2**(bits - 1)).astype("uint8")
Sref = (scales).astype(out_dtype)
Bref = activation
Zref = zero_points

if m_groups == -1:
    # Handle multi-channel case where scales/zeros shape is (1, K//group_size)
    if scales.shape[0] == 1:
        # Multi-channel quantization
        Adq = Aref.T.reshape(K // group_size, group_size,
                             M // bits).astype(out_dtype) - (2**(bits - 1))
        # Transpose to (group_size, K//group_size, M//bits)
        Adq = Adq.transpose(1, 0, 2)
        # Apply scales: Sref.T shape is (K//group_size, 1), broadcast to each group
        Adq = Adq * Sref.T.reshape(1, K // group_size, 1)
        if zero_point:
            # Apply zero_points (subtract, same as per-group case)
            Adq = Adq + Zref.T.reshape(1, K // group_size, 1)
        # Transpose back and reshape
        Adq = Adq.transpose(1, 0, 2).reshape(K, M // bits)
    else:
        # Original per-group quantization
        Adq = Aref.T.reshape(K // group_size, group_size,
                             M // bits).astype(out_dtype) - (2**(bits - 1))
        Adq = Adq.transpose(1, 0, 2) * Sref.T
        if zero_point:
            Adq = Adq - Zref.T
        Adq = Adq.transpose(1, 0, 2).reshape(K, M // bits)
else:
    Adq = (Aref.T.astype(out_dtype) - (2**(bits - 1))) * Sref[0]

# pesudo_qweight = qweight.reshape(M // bits, K // group_size, group_size).astype(out_dtype)
# pesudo_qweight = (pesudo_qweight.transpose(2, 1, 0) * scale.T).transpose(2, 1, 0).reshape(M // bits, K).astype(out_dtype)
# qweight = np.round(qweight + 2 ** (bits - 1)).astype("uint8")

Y_ref = weight.dot(activation.T)
Cref = Bref.dot(Adq)

# Compare with dequantized result
Y_dequant = activation.dot(dequantized_weight_mc.T)
print(f"\nComparison:")
print(f"  Y_ref vs Y_dequant NMSE: {nmse(Y_ref, Y_dequant):.6f}")
print(f"  Cref vs Y_dequant allclose: {np.allclose(Cref, Y_dequant)}")

__import__('pdb').set_trace()

# Expand scales and zero_points to [M//bits, K] from any shape
target_M = M // bits
target_K = K


# Function to expand any shaped array to target shape
def expand_to_shape(arr, target_shape):
    """Expand array to target shape by repeating elements."""
    current_shape = arr.shape
    target_M, target_K = target_shape

    # Calculate repeat factors for each dimension
    repeat_M = target_M // current_shape[0] if target_M % current_shape[
        0] == 0 else 1
    repeat_K = target_K // current_shape[1] if target_K % current_shape[
        1] == 0 else 1

    # Expand along M dimension if needed
    if current_shape[0] < target_M:
        arr = np.repeat(arr, repeat_M, axis=0)

    # Expand along K dimension if needed
    if current_shape[1] < target_K:
        arr = np.repeat(arr, repeat_K, axis=1)

    return arr


# Expand scales and zero_points
Sref_expanded = expand_to_shape(Sref, (target_M, target_K))
Zref_expanded = expand_to_shape(Zref, (target_M, target_K))

print(f"Original Sref shape: {Sref.shape}, Expanded: {Sref_expanded.shape}")
print(f"Original Zref shape: {Zref.shape}, Expanded: {Zref_expanded.shape}")

# Verify the expansion is correct
assert Sref_expanded.shape == (
    target_M, target_K
), f"Sref expansion failed: {Sref_expanded.shape} != {(target_M, target_K)}"
assert Zref_expanded.shape == (
    target_M, target_K
), f"Zref expansion failed: {Zref_expanded.shape} != {(target_M, target_K)}"

A_t, Scales_t = preprocess_weights(
    Aref,
    Sref_expanded,
    Zref_expanded,
    bits=bits,
    g=g,
    bm=bm,
    kfactor=kfactor,
    simd_n_in=simd_n_in,
    simd_n_out=simd_n_out,
)

# Debug output to verify shapes
print(f"A_t shape: {A_t.shape}")
print(f"Scales_t shape: {Scales_t.shape}")

#! ========================================================================================================================


def preprocessor_reference(B, act_group_size, g, dtype, out_dtype):
    _states = [-1, 1]
    _gamma = 1
    maxv = (1 << 7) - 1  # > 127, indicate range [-127, 127]

    b = B.reshape(N, K // g, g)

    # > generate
    codes = np.array([[i] for i in range(1 << g)], dtype=np.uint8)
    codes = np.unpackbits(codes, axis=1, bitorder="little", count=g).T

    def map_states(c):
        return _states[c]

    m = np.vectorize(map_states)(codes).astype(out_dtype)

    # (N, K // g, 1 << g)
    lut = b.dot(m)
    lut_biases = lut.reshape(N, K // act_group_size, act_group_size // g,
                             1 << g)[:, :, :,
                                     0]  # > pick minimum value of each group.
    lut_biases = np.sum(lut_biases, axis=-1) * _gamma

    # > Quantize LUT into int8, q_val between [-127, 127]
    qlut = lut.reshape(N, K // act_group_size, act_group_size // g * (1 << g))
    absmax = np.max(
        np.abs(qlut),
        axis=-1)  #! This abs + max will calculate the SUM of each group.
    lut_scales = absmax / maxv

    def recp(s):
        return 1.0 / s if s != 0 else 0

    ils = np.vectorize(recp)(lut_scales).astype(out_dtype)
    qlut = np.rint(
        (qlut.transpose(2, 0, 1).reshape(-1, qlut.shape[0] * qlut.shape[1]) *
         ils.reshape(1, qlut.shape[0] * qlut.shape[1])).reshape(
             qlut.shape[2], qlut.shape[0],
             qlut.shape[1]).transpose(1, 2, 0).reshape(N, K // g,
                                                       1 << g)).astype(dtype)

    return B, lut_scales, lut_biases, qlut


# 运行预处理
Bref, LUT_Scales, LUT_Biases, QLUT = preprocessor_reference(
    Bref, act_group_size, g, dtype, out_dtype)

#! ========================================================================================================================


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4, 8, 16, 32, 64]
    return alphas[:bits]


# 矩阵乘法部分
def qgemm_reference(
    A,
    QLUT,
    LUT_Scales,
    LUT_Biases,
    scales,
    bits,
    g,
    group_size,
    m_groups,
    simd_n_in,
    simd_n_out,
    bm=512,
    kfactor=16,
):
    """修改后的正确实现"""
    # 从预处理后的A获取维度信息
    M_bm, K_g, A_cols = A.shape  # 预处理后的A形状 (M//bm, K//g, bm//ngroups_per_elem)
    _ngroups_per_elem = 8 // g  # 每个uint8元素包含的组数
    # simd_n_in = 16               # 根据preprocess_weights中的参数
    # simd_n_out = 8               # 根据preprocess_weights中的参数

    # 计算实际维度
    M = M_bm * bm  # 原始权重行数
    K = K_g * g  # 原始输入维度
    N, _, lut_size = QLUT.shape  # QLUT形状 (N, K//g, 16)

    # 计算中间维度参数
    mgroup = _ngroups_per_elem * simd_n_in
    num_simd_blocks = bm // mgroup

    alphas = get_bits_alphas(bits)

    cbits = np.zeros((N, M), dtype=out_dtype)
    # 初始化累加器

    # > 这里其实相当于展开之前被合并起来的 排布 。
    A = A.reshape(
        M // bm,
        K // g // kfactor,
        bm // _ngroups_per_elem // simd_n_in,
        kfactor,
        simd_n_in,
    )
    A = np.concatenate([(A >> (g * ng)) & ((1 << g) - 1)
                        for ng in range(_ngroups_per_elem)],
                       axis=-1)

    # Check if scales are per-channel (shape includes K dimension)
    if scales.shape[1] == K:
        # Per-channel scales [M//bm, K, ...]
        if not zero_point:
            scales = scales.reshape(M // bm, K, bm // bits // simd_n_out,
                                    simd_n_out)
        else:
            scales = scales.reshape(M // bm, K, bm // bits // simd_n_out, 2,
                                    simd_n_out)
    else:
        # Original per-group scales [M//bm, K//group_size, ...]
        if not zero_point:
            scales = scales.reshape(M // bm, K // group_size,
                                    bm // bits // simd_n_out, simd_n_out)
        else:
            scales = scales.reshape(M // bm, K // group_size,
                                    bm // bits // simd_n_out, 2, simd_n_out)

    # import pdb; pdb.set_trace()
    for n in range(N):
        for k in range(K // g):
            for m in range(M):
                # NOTE: A shape is (M // bm, K // g // kfactor, bm // _ngroups_per_elem // simd_n_in, kfactor, simd_n_in)

                # > In this Loop, M is M * bits.
                mo = m // bm
                ko = k // kfactor
                mi = (m % bm) // _ngroups_per_elem // simd_n_in
                ki = k % kfactor
                e = (m % bm) % (_ngroups_per_elem * simd_n_in)
                a_e = A[mo, ko, mi, ki, e]  # > Get the index of weight.

                scales_mi = (m % bm) // bits // simd_n_out
                scales_e = (m % bm) % simd_n_out

                # import pdb; pdb.set_trace()

                if m_groups == -1:
                    # Check if per-channel or per-group scales
                    if scales.shape[1] == K:
                        # Per-channel scales
                        if zero_point:
                            s = scales[mo, k * g, scales_mi, 0, scales_e]
                        else:
                            s = scales[mo, k * g, scales_mi, scales_e]
                    else:
                        # Per-group scales
                        if zero_point:
                            s = scales[mo, k * g // group_size, scales_mi, 0,
                                       scales_e]
                        else:
                            s = scales[mo, k * g // group_size, scales_mi,
                                       scales_e]
                else:
                    m_group_size = M // m_groups
                    s = scales[m // m_group_size]

                cbits[n, m] += (QLUT[n, k, a_e] *
                                LUT_Scales[n, k * g // act_group_size] * s)
                if (((k * g) % act_group_size)
                        == 0) and ((((m % bm) // simd_n_out) % bits) == 0):
                    cbits[n, m] += LUT_Biases[n, k * g // act_group_size] * s
                    if zero_point:
                        # Check if per-channel or per-group scales for zero_point
                        if scales.shape[1] == K:
                            zp = scales[mo, k * g, scales_mi, 1, scales_e]
                        else:
                            zp = scales[mo, k * g // group_size, scales_mi, 1,
                                        scales_e]
                        cbits[n,
                              m] += (-LUT_Biases[n, k * g // act_group_size] *
                                     (1 / alphas[0]) * zp)

    c = (cbits.reshape(
        (N, M // simd_n_out // bits, bits,
         simd_n_out)).transpose(0, 1, 3,
                                2).dot(np.array(alphas,
                                                dtype=out_dtype)).reshape(
                                                    (N, M // bits)))

    return c


C = qgemm_reference(
    A_t,
    QLUT,
    LUT_Scales,
    LUT_Biases,
    Scales_t,
    bits,
    g,
    group_size,
    m_groups,
    simd_n_in=simd_n_in,
    simd_n_out=simd_n_out,
    bm=bm,
    kfactor=kfactor,
)

print("\n" + "=" * 60)
print("Results comparison:")
print("=" * 60)
print(f"Y_ref (FP16) first 10 elements: {Y_ref.flatten()[:10]}")
print(f"Cref (Dequantized) first 10 elements: {Cref.flatten()[:10]}")
print(f"C (LUT) first 10 elements: {C.flatten()[:10]}")

# Calculate various error metrics
NMSE_ref_to_dequant = nmse(Y_ref, Cref)
NMSE_dequant_to_lut = nmse(Cref, C)

print(f"\nError Metrics:")
print(f"  Y_ref vs Cref NMSE: {NMSE_ref_to_dequant:.6f}")
print(f"  Cref vs C (LUT) NMSE: {NMSE_dequant_to_lut:.6f}")
print(f"  Cref vs C allclose: {np.allclose(Cref, C, rtol=1e-3)}")

# Check if Adq matches dequantized weight
Y_check = activation.dot(dequantized_weight_mc.T)
print(f"\nVerification:")
print(
    f"  Cref equals activation.dot(dequantized_weight.T): {np.allclose(Cref, Y_check)}"
)
print(f"  Max difference between Cref and C: {np.max(np.abs(Cref - C)):.6f}")
