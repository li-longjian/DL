import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35

#train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
train_iter, vocab = d2l.load_data_voc(batch_size, num_steps) ##数据集有太大。。。

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    '''
    W_xz 和 W_hz 是更新门的输入和隐藏状态的权重。
    b_z 是更新门的偏置。
    W_xr 和 W_hr 是重置门的输入和隐藏状态的权重。
    b_r 是重置门的偏置。
    W_xh 和 W_hh 是候选隐藏状态的输入和隐藏状态的权重。
    b_h 是候选隐藏状态的偏置。
    W_hq 和 b_q 是输出层的权重和偏置
    '''
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state #从状态元组 state 中提取出隐藏状态 H
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z) #计算更新门 Z
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)#计算重置门 R
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h) #计算候选隐藏状态 H_tilda
        H = Z * H + (1 - Z) * H_tilda #更新隐藏状态 H
        Y = H @ W_hq + b_q #计算当前时间步的输出 Y
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)