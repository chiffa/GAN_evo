# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : config.py
# @Time         : Created at 2019-03-18
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.
import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import config as cfg
from utils.helpers import truncated_normal_


class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'pineapple'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_params()

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]

        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=cfg.batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c


class TransformerGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, num_heads=4, nlayers=3, dropout=0.5, gpu=False):
        super(TransformerGenerator, self).__init__()
        self.model_type = 'TransformerGenerator'
        # Compared to pytorch transformer_tutorial: ntoken = vocab_size    and  ninp= embedding_dim  nhid=hidden_dim

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        print(f"embed_dim: {embedding_dim}, hidden_dim: {hidden_dim}, num_heads:{num_heads}")
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.decoder = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        output = self.softmax(output)
        return output

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out = self.forward(inp, self.generate_square_subsequent_mask(self.max_seq_len))  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]
        return samples

    #Replace init_hidden, not sure it's working properly
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)