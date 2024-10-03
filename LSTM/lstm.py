import torch
from torch import nn
from d2l import torch as d2l #H.G.Wells的时光机器数据集

batch_size, num_steps = 32, 35
#train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

### 定义和初始化模型参数
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        ##在指定的 device 上生成一个形状为 shape 的随机张量，这个张量中的值服从标准正态分布。
        ##将生成的随机张量中的每个值乘以 0.01。这通常是为了初始化参数时避免过大的值，以免在训练过程中导致梯度爆炸等问题
        return torch.randn(size=shape, device=device)*0.01

    def three():##返回一个包含三个张量的元组。这三个张量分别用于初始化 LSTM 的不同部分
        return (normal((num_inputs, num_hiddens)),
                ##生成一个形状为 (num_inputs, num_hiddens) 的随机张量，可能用于输入到 LSTM 门控单元的权重初始化
                normal((num_hiddens, num_hiddens)),
                ##生成一个形状为 (num_hiddens, num_hiddens) 的随机张量，可能用于隐藏状态到门控单元的权重初始化。
                torch.zeros(num_hiddens, device=device))
                ##在指定设备上生成一个形状为 (num_hiddens,) 的全零张量，可能用于门控单元的偏置项初始化。

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))##生成一个形状为 (num_hiddens, num_outputs) 的随机张量，用于从隐藏状态到输出的权重初始化。
    b_q = torch.zeros(num_outputs, device=device)#形状为 (num_outputs,) 的全零张量，作为输出层的偏置项

    # 附加梯度
    ##将所有初始化的参数收集到一个列表 params 中。这个列表包含了 LSTM 模型的所有可训练参数
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True) ## True:意味着在模型训练过程中，这些参数的梯度将被计算，以便通过反向传播算法进行优化
    return params

##给定批大小和隐藏状态维度的 LSTM 模型提供初始状态，以便在模型的前向传播过程中开始处理数据。
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),##初始隐藏状态（h_0）
            torch.zeros((batch_size, num_hiddens), device=device))##初始细胞状态（c_0）
###返回一个包含两个张量的元组。这两个张量分别表示 LSTM 的初始隐藏状态（h_0）和初始细胞状态（c_0）

def lstm(inputs, state, params):
    ##input:形状为 (sequence_length, batch_size, input_size)
    ##state：表示 LSTM 的初始状态，是一个包含两个张量的元组，分别是隐藏状态 H 和细胞状态 C。
    ##params：LSTM 的参数列表，通常包括各种权重矩阵和偏置项。

    ##拆分成各个具体的参数，分别对应 LSTM 的输入门、遗忘门、输出门、候选记忆元以及输出层的权重矩阵和偏置项
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    #将传入的初始状态 state 拆分成隐藏状态 H 和细胞状态 C。
    (H, C) = state
    outputs = []
    for X in inputs:
        ##计算输入门的值 I
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        ##计算遗忘门的值 F
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        ##计算输出门的值 O
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        ##候选细胞状态
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        ##细胞状态
        C = F * C + I * C_tilda
        ##隐藏状态
        H = O * torch.tanh(C)
        ##当前时间步输出
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    ##将 outputs 列表中的所有时间步的输出沿着指定维度（这里是维度 0，通常对应时间步维度）拼接起来，得到最终的输出序列
    return torch.cat(outputs, dim=0), (H, C)


