import torch

# Define float input
x = torch.rand(1000, 1000) * torch.randn(1000, 1000)


def int_softmax(x):
    # Quantize the input, asymmetric quantization. Note this input quantization error is independent of softmax computation error
    scaling_factor = (torch.max(x) - torch.min(x)) / 255
    zero = -128 - torch.min(x) / scaling_factor
    x_int = torch.clip(torch.round(x / scaling_factor + zero), -128, 127)
    ref_out = torch.nn.functional.softmax((x_int - zero) * scaling_factor, -1)

    # Subtract maximum to prevent overflow
    x_int_max, _ = torch.max(x_int, dim=-1, keepdim=True)
    x_int = x_int - x_int_max  # x_int is 8bit with values in [-255, 0]

    # NOTE: This is main optimize part of int softmax.
    Ip = x_int + torch.floor(x_int / 2) - torch.floor(x_int / 2**4)  # int16
    Ip = torch.clamp(Ip, -2**15, 2**15 - 1)

    if scaling_factor < 1 / 255:
        r = Ip
        y = torch.floor(r / 2 - torch.floor(-1.0 / scaling_factor))
        y = torch.clamp(y, -2**31, 2**31 - 1)
        n = torch.clamp(torch.floor(torch.log2(y.max())) - 6, 0)
        y = y / 2**n
        exp_int = torch.clamp(y, -2**7, 2**7 - 1)
    elif scaling_factor >= 1:
        q = -Ip
        exp_int = 2**-q
    else:
        q = torch.floor(Ip /
                        torch.floor(-1.0 / scaling_factor))  # q is positive
        q = torch.clamp(q, -2**15, 2**15 - 1)
        r = Ip - torch.floor(-1.0 / scaling_factor) * q  # Here r is negative
        r = torch.clamp(r, -2**15, 2**15 - 1)

        y = torch.floor(r / 2 - torch.floor(-1.0 / scaling_factor))
        y = torch.clamp(y, -2**15, 2**15 - 1)

        # How to set n here also needs further research
        n = -torch.floor(torch.log2(128 / y.max()))
        q = q + n

        exp_int = torch.floor(y / 2**q)
        exp_int = torch.clamp(exp_int, -2**7, 2**7 - 1)

    exp_int = exp_int.float()
    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
    out = exp_int / exp_int_sum

    # Quantize the output, asymmetric quantization. Note this output quantization error is independent of softmax computation error
    scaling_factor = (torch.max(out) - torch.min(out)) / 255
    zero = -128 - torch.min(out) / scaling_factor
    out_int = torch.clip(torch.round(out / scaling_factor + zero), -128, 127)
    out = (out_int - zero) * scaling_factor
    return out, ref_out


def int_softmax_ibert(x):
    # Quantize the input, asymmetric quantization. Note this input quantization error is independent of softmax computation error
    scaling_factor = (torch.max(x) - torch.min(x)) / 255
    zero = -128 - torch.min(x) / scaling_factor
    x_int = torch.clip(torch.round(x / scaling_factor + zero), -128, 127)
    ref_out = torch.nn.functional.softmax((x_int - zero) * scaling_factor, -1)

    # Subtract maximum to prevent overflow
    x_int_max, _ = torch.max(x_int, dim=-1, keepdim=True)
    x_int = x_int - x_int_max  # x_int is 8bit with values in [-255, 0]

    if scaling_factor >= 2 * torch.log(torch.tensor(
            2.)):  # Special case: quantization factor too large, causing r=0
        # Calculate z
        z = torch.floor(
            x_int / torch.floor(-torch.log(torch.tensor(2.)) / scaling_factor))
        z = torch.clamp(z, -128, 127)  # z is 8bit signed integer
        exp_int = 2**-z

    else:
        # Calculate z and r. When input magnitude range is large, z_max approaches 255 and r approaches 0. When input magnitude range is small, r approaches -255 and z approaches 0
        z = torch.floor(
            x_int / torch.floor(-torch.log(torch.tensor(2.)) / scaling_factor))
        z = torch.clamp(
            z, -128, 127)  # z is 8bit signed integer, but must be non-negative

        r = x_int - torch.floor(
            -torch.log(torch.tensor(2.)) / scaling_factor) * z
        r = torch.clamp(
            r, -128, 127)  # r is 8bit signed integer, but must be non-positive

        # Calculate exp()
        A = 0.35815147
        B = 0.96963238
        C = 1.
        b_int = torch.floor(B / scaling_factor / A)
        c_int = torch.floor(C / scaling_factor**2 / A)
        b_int = torch.clamp(b_int, -2**31,
                            2**31 - 1)  # b_int is 32bit signed integer
        c_int = torch.clamp(c_int, -2 * 31,
                            2**31 - 1)  # c_int is 32bit signed integer

        y = r * (r + b_int) + c_int
        y = torch.clamp(y, -2**31, 2**31 - 1)
        y_max = c_int
        # The introduction of n here is crucial and deserves further research on what value is most reasonable!
        n = torch.clamp(torch.floor(torch.log2(y_max)) - 6, 0)
        z = z + n

        out = y.to(torch.int32) >> z.to(torch.int32)
        exp_int = torch.clamp(out, -2**7, 2**7 - 1)

    exp_int = exp_int.float()
    exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
    out = exp_int / exp_int_sum

    # Quantize the output, asymmetric quantization. Note this output quantization error is independent of softmax computation error
    scaling_factor = (torch.max(out) - torch.min(out)) / 255
    zero = -128 - torch.min(out) / scaling_factor
    out_int = torch.clip(torch.round(out / scaling_factor + zero), -128, 127)
    out = (out_int - zero) * scaling_factor
    return out, ref_out


def compute_nmse(out, ref_out):
    """
    Compute normalized mean square error
    """
    return torch.mean((out - ref_out)**2) / torch.mean(ref_out**2)


out, ref_out = int_softmax(x)
out_ibert, ref_out_ibert = int_softmax_ibert(x)

print("NMSE:", compute_nmse(out, ref_out).item())
print("NMSE ibert:", compute_nmse(out_ibert, ref_out_ibert).item())

__import__('pdb').set_trace()

print(torch.nn.functional.cosine_similarity(ref_out, out, -1).mean())
