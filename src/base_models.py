import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class LinearRegressionModel(nn.Module):
    def __init__(self, in_size=50,  out_size=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
    
    def forward(self, x):
        return self.linear(x)

# 平方线性回归模型
class QuadraticRegressionModel(nn.Module):
    def __init__(self, in_size=50, out_size=1):
        super(QuadraticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
    
    def forward(self, x):
        return self.linear(x ** 2)
    
class LinearQuadraticRegressionModel(nn.Module):
    def __init__(self, in_size=50, out_size=1):
        super(LinearQuadraticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size, bias=False)
        self.linear2 = nn.Linear(in_size, out_size, bias=False)
    
    def forward(self, x):
        return self.linear1(x) + self.linear2(x ** 2)
    
class PowerRegressionModel(nn.Module):
    def __init__(self, in_size=50,  out_size=1, power=2.0):
        super(PowerRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.power=power
    def forward(self, x):
        return self.linear(x**self.power)



class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for i in range(num_models)]
        )

    def forward(self, xs):
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs
