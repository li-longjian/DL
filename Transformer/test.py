import math

import numpy as np
import torch
from torch import nn
##8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels= d_ff,kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff,out_channels=d_model,kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.tranpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        return self.layer_norm(output + residual)

## 7. ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(d_k)

        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

## 6. MutiHeadAttention
class MutiHeadAttention(nn.Module):
    def __init__(self):
        super(MutiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads,d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

## 5.EncoderLayer 包含多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MutiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


## 4.get_attn_pad_mask
def get_attn_pad_mask(seq_q, seq_k):
     batch_size, len_q = seq_q.size()
     batch_size, len_k = seq_k.size()
     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
     return pad_attn_mask.expand(batch_size,len_q,len_k)

# 3. PositionalEncoding 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pe = troch.arra
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose()

        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#2. Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ##a.词嵌入
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        ## b.位置编码
        self.pos_emb = PositionalEncoding(d_model)

        ## c.
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    def forward(self,enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1)

        ## 得到填充掩码位置
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs,enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
##10. get_attn_subsequent_mask
def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    subsquence_mask = np.triu(np.ones(attn_shape),k=1)
    subsquence_mask = torch.from_numpy(subsquence_mask)
    return subsquence_mask;

## 11.DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MutiHeadAttention()
        self.dec_enc_attn = MutiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, dec_inputs, enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
## 9. Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask((dec_inputs,dec_inputs))

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns



#1.Tansformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model,tgt_vocab_size,bias=False)

    def forward(self,enc_inputs,dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns  = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1,dec_logits.size(-1)), enc_self_attns,dec_self_attns,dec_enc_attns


if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    ##构建词表
    src_vocab = {'P':0,'ich':1,'mochte':2,'ein':3,'bier':4}
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5

    ## 初始化参数
    d_model = 512  ## Embedding size
    d_ff = 2048 ## 前馈神经网络维度
    d_k = d_v = 64 ## QKV的维度
    n_layers = 6 ## number layer encoder and decoder
    n_heads = 8 # number of heads in Multi-Head Attention

    model = Transformer()