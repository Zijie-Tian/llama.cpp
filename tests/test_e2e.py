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

    if scales.size >= M // bits:
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

bits = 4
M = 128 * bits
N = 1
K = 128  # > K >= g * kfactor && K >= act_group_size && K >= group_size
g = 4
bm = 128 * bits
simd_n_in = 16
simd_n_out = 8
kfactor = 16
act_group_size = 4  # > act_group_size >= g
group_size = 128
out_dtype = "float16"
dtype = "int8"
zero_point = False
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


# weight = np.random.randn(M // bits, K).astype(out_dtype)      # FP16
weight = np.ones((M // bits, K)).astype(out_dtype)
activation = np.random.randn(N, K).astype(out_dtype)  # FP16

weight_quant, scales = quantize_weight_per_tensor(weight, bits=bits)
print("weight_quant:", weight_quant)
print("scales:", scales)

dequantized_weight = dequantize_weight_per_tensor(weight_quant, scales)
print("dequantized_weight:", dequantized_weight)

qweight = weight_quant
scale = scales * np.ones((M // bits, K // group_size))
Bref = activation

# weight = np.load("weight.npy").astype(out_dtype)
# activation = np.load("activation.npy").astype(out_dtype) # FP16

# TODO : Implement real group quantization.
# qweight, scale = weight_quant_numpy(weight, -1)
# scale = scale * np.ones((M // bits, K // group_size), dtype=out_dtype)

Aref = np.round(qweight + 2**(bits - 1)).astype("uint8")
Sref = (scale * np.ones((M // bits, K // group_size))).astype(out_dtype)
Bref = activation
Zref = None

if m_groups == -1:
    Adq = Aref.T.reshape(K // group_size, group_size,
                         M // bits).astype(out_dtype) - (2**(bits - 1))
    # > [group_size, K // group_size, M // bits] * [K // group_size, M // bits]
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

A_t, Scales_t = preprocess_weights(
    Aref,
    Sref,
    None,
    bits=bits,
    g=g,
    bm=bm,
    kfactor=kfactor,
    simd_n_in=simd_n_in,
    simd_n_out=simd_n_out,
)

__import__("pdb").set_trace()

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

    #! 注意这地方其实没啥问题，因为TMAC应该没有对weight进行group的量化，是per tensor的量化。
    scales = scales.reshape(M // bm, K // group_size, bm // bits // simd_n_out,
                            simd_n_out)

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
                        cbits[n,
                              m] += (LUT_Biases[n, k * g // act_group_size] *
                                     (1 / alphas[0]) *
                                     scales[mo, k * g // group_size, scales_mi,
                                            1, scales_e])

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

print("Non-quantized Y_ref:", Y_ref.flatten())
print("Reference C:", Cref.flatten())
print("Simulated C:", C.flatten())

NMSE = nmse(Cref, C)

print("NMSE :", NMSE)
