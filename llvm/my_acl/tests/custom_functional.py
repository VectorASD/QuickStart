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



def geglu_torch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    GELU-Gated Linear Unit.
    Вход: тензор формы [..., 2*H]
    Выход: тензор формы [..., H]
    """
    # Разделяем на две половины по последней оси
    a, b = input_tensor.chunk(2, dim=-1)
    # Применяем GELU (tanh-приближение) к первой половине и умножаем на вторую
    return F.gelu(a, approximate="tanh") * b

def dgeglu_torch(grad_output: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Обратный проход для geglu.
    Вход: grad_output той же формы, что выход geglu,
          input_tensor исходный вход geglu.
    Выход: градиент по input_tensor.
    """
    # Прямой проход: geglu(x) = gelu(a) * b, где a, b = x.chunk(2, dim=-1)
    a, b = input_tensor.chunk(2, dim=-1)
    a = a.requires_grad_(True)
    gelu_a = F.gelu(a, approximate='tanh')
    y = gelu_a * b

    # Обратный проход: считаем градиент 'a' через autograd (aclnnGeluBackwardV2, сделанный через at::gelu_backward)
    grad_a = torch.autograd.grad(outputs=y, inputs=a, grad_outputs=grad_output, retain_graph=False)[0]
    grad_b = grad_output * gelu_a

    result = torch.cat((grad_a, grad_b), dim=-1)
    return result

torch.geglu = geglu_torch
torch.dgeglu = dgeglu_torch
torch.Tensor.geglu = lambda self: geglu_torch(self)
torch.Tensor.dgeglu = lambda self, grad_output: dgeglu_torch(grad_output, self)
