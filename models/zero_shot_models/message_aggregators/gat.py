import dgl.function as fn
import torch
from dgl.nn import edge_softmax
from dgl.utils import expand_as_pair
from torch import nn

from models.zero_shot_models.message_aggregators.aggregator import MessageAggregator


class GATConv(MessageAggregator):
    """
    A message aggregator that combines child messages using an attention mechanism similar to the GAT graph convolution.
    """

    def __init__(self,
                 hidden_dim=0,
                 num_heads=4,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 **fc_out_kwargs):
        super().__init__(input_dim=(num_heads + 1) * hidden_dim, output_dim=hidden_dim, **fc_out_kwargs)
        in_feats = hidden_dim
        out_feats = hidden_dim

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph=None, etypes=None, in_node_types=None, out_node_types=None, feat_dict=None):
        with graph.local_scope():
            graph.ndata['h'] = feat_dict
            feat_src = {
                t: self.fc(self.feat_drop(feat_dict[t])).view(-1, self._num_heads, self._out_feats)
                for t in in_node_types
            }
            feat_dst = {
                t: self.fc(self.feat_drop(feat_dict[t])).view(-1, self._num_heads, self._out_feats)
                for t in out_node_types
            }
            el = {t: (feat_src[t] * self.attn_l).sum(dim=-1).unsqueeze(-1)
                  for t in in_node_types}
            er = {t: (feat_dst[t] * self.attn_r).sum(dim=-1).unsqueeze(-1)
                  for t in out_node_types}

            for etype in etypes:
                t_in, edge_t, t_out = etype
                graph[etype].srcdata.update({'ft': feat_src[t_in], 'el': el[t_in]})
                graph[etype].dstdata.update({'er': er[t_out]})

            # previously
            # graph.srcdata.update({'ft': feat_src, 'el': el})
            # graph.dstdata.update({'er': er})

            for etype in etypes:
                graph.apply_edges(fn.u_add_v('el', 'er', 'e'), etype=etype)

            e_dict = graph.edata.pop('e')
            attention_dict = dict()
            for etype, e_data in e_dict.items():
                e = self.leaky_relu(e_data)
                # compute softmax
                edge_sm = self.attn_drop(edge_softmax(graph[etype], e))
                attention_dict[etype] = edge_sm
            graph.edata['a'] = attention_dict

            # message passing
            # graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
            #                 fn.sum('m', 'ft'), etype=f'E{depth}')
            graph.multi_update_all({etype: (fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) for etype in etypes},
                                   cross_reducer='sum')

            feat = graph.ndata['h']
            rst = {t: ndata.view(-1, self._num_heads * self._in_src_feats) for t, ndata in graph.dstdata['ft'].items()}

            out_dict = self.combine(feat, out_node_types, rst)

            return out_dict
