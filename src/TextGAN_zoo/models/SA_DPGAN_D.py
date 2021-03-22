import torch
import torch.nn.functional as F

import config as cfg
from models.generator import TransformerEncoder
from utils.data_loader import GenDataIter


class SA_DPGAN_D(TransformerEncoder):
    def __init__(self, embedding_dim, nhead, nhid, nlayers, dropout=0.5, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SA_DPGAN_D, self).__init__(self, embedding_dim, nhead, nhid, nlayers, dropout, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'sa_dpgan_d'

    def getReward(self, samples):
        """
        Get word-level reward and sentence-level reward of samples.
        """
        batch_size, _ = samples.size()
        inp, target = GenDataIter.prepare(samples, cfg.CUDA)

        src_mask = self.generate_square_subsequent_mask(self.max_seq_len)
        pred = self.forward(inp, src_mask)

        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)

        return word_reward, sentence_reward