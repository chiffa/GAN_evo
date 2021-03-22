import torch
import torch.nn.functional as F

from models.generator import TransformerEncoder


class SA_DPGAN_G(TransformerEncoder):
    def __init__(self, embedding_dim, nhead, nhid, nlayers, dropout=0.5, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SA_DPGAN_G, self).__init__(self, embedding_dim, nhead, nhid, nlayers, dropout, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'sa_dpgan_g'

    def sample_teacher_forcing(self, inp):
        """
        Generating samples from the real data via teacher forcing
        :param inp: batch_size * seq_len
        :param target: batch_size * seq_len
        :return
            samples: batch_size * seq_len
            log_prob: batch_size * seq_len  (log probabilities)
        """
        batch_size, _ = inp.size()
        src_mask = self.generate_square_subsequent_mask(self.max_seq_len)

        pred = self.forward(inp, src_mask)
        samples = torch.argmax(pred, dim=-1).view(batch_size, -1)
        log_prob = F.nll_loss(pred, samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)

        return samples, log_prob