import numpy as np
import paddle
import paddle.nn as nn

class FogPassFilter_conv1(nn.Layer):
    def __init__(self, inputsize):
        super(FogPassFilter_conv1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize // 2)
        self.hidden2 = nn.Linear(inputsize // 2, inputsize // 4)
        self.output = nn.Linear(inputsize // 4, 64)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)

        return x

class FogPassFilter_res1(nn.Layer):
    def __init__(self, inputsize):
        super(FogPassFilter_res1, self).__init__()
        
        self.hidden = nn.Linear(inputsize, inputsize // 8)
        self.output = nn.Linear(inputsize // 8, 64)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        
        return x

# Example usage
if __name__ == '__main__':
    inputsize = 256
    model_conv1 = FogPassFilter_conv1(inputsize)
    model_res1 = FogPassFilter_res1(inputsize)
    
    x = paddle.randn([1, inputsize], dtype='float32')
    
    output_conv1 = model_conv1(x)
    output_res1 = model_res1(x)
    
    print("Output from FogPassFilter_conv1:", output_conv1.numpy())
    print("Output from FogPassFilter_res1:", output_res1.numpy())
