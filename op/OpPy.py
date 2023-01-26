from torch.autograd import Function


class fused_leaky_relu_PY_Function(Function):

    @staticmethod
    def fused_leaky_relu_PY(self):
        return 0
