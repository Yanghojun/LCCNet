import correlation_cuda
import torch
from torch.autograd import Function
from torch.nn.modules.module import Module


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx, self, input1, input2):
    # def forward(ctx, self, input1, input2):     # ctx: Tensor임. forward 함수 호출할 때 Tensor를 인자로 넣잖아. 그게 이것임.
        ctx.self = self
        ctx.save_for_backward(input1, input2)
        # ctx.save_for_backward(self, input1, input2)
        # self.save_for_backward(input1, input2)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

            # correlation_cuda.forward(input1, input2, rbot1, rbot2, output, 
            #     0, 0, 0, 1, 2, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):     # ctx는 오차역전파를 위한 값인듯
        # self, input1, input2 = ctx.saved_tensors
        input1, input2 = ctx.saved_tensors
        self = ctx.self

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

            # correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
            #     0, 0, 0, 1, 2, 1)

        return None, grad_input1, grad_input2


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)
        
        # result = result.apply(self, input1, input2)
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        result = CorrelationFunction.apply(self, input1, input2)

        return result