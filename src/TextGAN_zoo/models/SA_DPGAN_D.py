import torch
import torch.nn.functional as F

import config as cfg
from models.generator import TransformerGenerator
from utils.data_loader import GenDataIter


class SA_DPGAN_D(TransformerGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, num_heads=4, nlayers=4, dropout=0.5, gpu=False):
        super(SA_DPGAN_D, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, num_heads, nlayers, dropout, gpu)
        self.name = 'sa_dpgan_d'

    def getReward(self, samples):
        """
        Get word-level reward and sentence-level reward of samples.
        """
        batch_size, _ = samples.size()
        inp, target = GenDataIter.prepare(samples, cfg.CUDA) # [batch_size, max_seq_len], [batch_size, max_seq_len]
        inp = inp.transpose(1, 0)       # [max_seq_len, batch_size]
        target = target.transpose(1, 0) # [max_seq_len, batch_size]

        src_mask = self.generate_square_subsequent_mask(self.max_seq_len)
        pred = self.forward(inp, src_mask)

        word_reward = F.nll_loss(pred, target.contiguous().view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)

        return word_reward, sentence_reward