import torch
from torch import nn


class EmbeddingInitializer(nn.Module):
    """
    Wrapper to generate a learnable embedding. Used whenever a categorical variable should be represented in zero-shot
    models.
    """

    def __init__(self, num_embeddings, max_emb_dim, p_dropout, minimize_emb_dim=True, drop_whole_embeddings=False,
                 one_hot=False):
        """
        :param minimize_emb_dim:
            Whether to set embedding_dim = max_emb_dim or to make embedding_dim smaller is num_embeddings is small
        :param drop_whole_embeddings:
            If True, dropout pretends the embedding was a missing value. If false, dropout sets embed features to 0
        :param drop_whole_embeddings:
            If True, one-hot encode variables whose cardinality is < max_emb_dim. Also, set reqiures_grad = False
        """
        super().__init__()
        self.p_dropout = p_dropout
        self.drop_whole_embeddings = drop_whole_embeddings
        if minimize_emb_dim:
            self.emb_dim = min(max_emb_dim, num_embeddings)  # Don't use a crazy huge embedding if not needed
        else:
            self.emb_dim = max_emb_dim
        # Note: if you change the name of self.embed, or initialize an embedding elsewhere in a model,
        # the function get_optim_no_wd_on_embeddings will not work properly
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.emb_dim)
        self.embed.weight.data.clamp_(-2, 2)  # fastai uses truncated normal init
        if one_hot:
            self.embed.weight.requires_grad = False
            if num_embeddings <= max_emb_dim:
                torch.eye(self.emb_dim, out=self.embed.weight.data)
        self.do = nn.Dropout(p=p_dropout)

    def forward(self, input):
        if self.drop_whole_embeddings and self.training:
            mask = torch.zeros_like(input).bernoulli_(1 - self.p_dropout)
            input = input * mask
        out = self.embed(input)
        if not self.drop_whole_embeddings:
            out = self.do(out)
        return out
