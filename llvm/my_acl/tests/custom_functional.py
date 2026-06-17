import torch
import torch.nn.functional as F



def broadcast_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(a.shape, b.shape)
    a = a.expand(shape).contiguous()
    b = b.expand(shape).contiguous()
    return a, b


_bitwise_left_shift = torch.bitwise_left_shift
_lshift = torch._C._TensorBase.__lshift__

def lshift(L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    if L.device.type == "cpu":
        return _bitwise_left_shift(L, R)
    L, R = broadcast_tensors(L, R)
    return _lshift(L, R)
    # [W610 18:26:03.055846726 VariableFallbackKernel.cpp:250] Warning: CAUTION: The operator 'aten::bitwise_left_shift.Tensor_out' is not currently supported on the NPU backend and will fall back to run on the CPU. This may have performance implications. (function npu_cpu_fallback)
    # И кто эту шляпу придумал (недодумал)?! Зато оператор '<<' ещё как не fallback'ается.
    # Ещё и внутри op-plugin накосячили, что требуется теперь broadcast_tensors из-за недопустимного прямого expand внутри -_-

torch.bitwise_left_shift = lshift
torch.Tensor.bitwise_left_shift = lambda self, other: lshift(self, other)
torch.Tensor.bitwise_left_shift_ = lambda self, other: self.copy_(lshift(self, other))
torch.Tensor.__lshift__ = lambda self, other: lshift(self, other)
torch.Tensor.__ilshift__ = lambda self, other: self.copy_(lshift(self, other))


_bitwise_right_shift = torch.bitwise_right_shift
_rshift = torch._C._TensorBase.__rshift__

def rshift(L: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    if L.device.type == "cpu":
        return _bitwise_right_shift(L, R)
    L, R = broadcast_tensors(L, R)
    return _rshift(L, R)

torch.bitwise_right_shift = rshift
torch.Tensor.bitwise_right_shift = lambda self, other: rshift(self, other)
torch.Tensor.bitwise_right_shift_ = lambda self, other: self.copy_(rshift(self, other))
torch.Tensor.__rshift__ = lambda self, other: rshift(self, other)
torch.Tensor.__irshift__ = lambda self, other: self.copy_(rshift(self, other))



def cpu_geglu(input_tensor: torch.Tensor, dim: int = -1, approximate: int = 1, activate_left: bool = False) -> torch.Tensor:
    """
    GELU-Gated Linear Unit.
    Вход: тензор формы [..., 2*H]
    Выход: тензор формы [..., H]
    """
    if activate_left:
        a, b = input_tensor.chunk(2, dim=dim)
    else:
        b, a = input_tensor.chunk(2, dim=dim)
    gelu = F.gelu(a, approximate="tanh" if approximate else "none")
    return gelu * b, gelu

def cpu_geglu_grad(grad_output: torch.Tensor,
                   input_tensor: torch.Tensor,
                   gelu: torch.Tensor = None,
                   dim: int = -1,
                   approximate: int = 1,
                   activate_left: bool = False) -> torch.Tensor:
    """
    CPU-референс для npu_geglu_grad.
    Параметры:
        grad_output: градиент от вышестоящей операции, форма [..., H]
        input_tensor: исходный вход geglu, форма [..., 2*H]
        gelu: промежуточный результат GELU (gelu(a)) той же формы, что и grad_output
        dim: ось, по которой делится input_tensor
        approximate: 1 для tanh-GELU, иначе 0 для стандартного
        activate_left: если False, GELU был применён к правой половине (как в npu_geglu по умолчанию)
    Возвращает: градиент по input_tensor, форма [..., 2*H]
    """
    if activate_left:
        a, b = input_tensor.chunk(2, dim=dim)
    else:
        b, a = input_tensor.chunk(2, dim=dim)

    # Градиент по a: используем gelu_backward (как в GeGluV3Backward)
    grad_a = torch.ops.aten.gelu_backward(grad_output * b, a, approximate='tanh' if approximate else 'none')
    grad_b = grad_output * gelu

    if activate_left:
        return torch.cat((grad_a, grad_b), dim=dim)
    return torch.cat((grad_b, grad_a), dim=dim)

torch.cpu_geglu = cpu_geglu
torch.cpu_geglu_grad = cpu_geglu_grad
torch.Tensor.cpu_geglu = lambda self: cpu_geglu(self)
torch.Tensor.cpu_geglu_grad = lambda self, grad_output: cpu_geglu_grad(grad_output, self)
