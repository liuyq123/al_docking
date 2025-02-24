from typing import Optional, List

import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dgl
import dgllife

class GCNModel(nn.Module):
    def __init__(self,
                 graph_conv_layers: List[int],
                 activation=None,
                 residual: bool = True,
                 batchnorm: bool = False,
                 dropout: float = 0.,
                 predictor_hidden_feats: int = 128,
                 predictor_dropout: float = 0.,
                 number_atom_features: int = 30):

        super(GCNModel, self).__init__()

        num_gnn_layers = len(graph_conv_layers)

        if activation is not None:
            activation = [activation] * num_gnn_layers
        
        self.model = dgllife.model.GCNPredictor(
            in_feats=number_atom_features,
            hidden_feats=graph_conv_layers,
            activation=activation,
            residual=[residual] * num_gnn_layers,
            batchnorm=[batchnorm] * num_gnn_layers,
            dropout=[dropout] * num_gnn_layers,
            n_tasks=1,
            predictor_hidden_feats=predictor_hidden_feats,
            predictor_dropout=predictor_dropout)

    def forward(self, g):
        out = self.model(g, g.ndata['x'])

        return out

class GATModel(nn.Module):
    def __init__(self,
                 graph_attention_layers: List[int],
                 n_attention_heads: int = 8,
                 agg_modes: Optional[list] = None,
                 activation=None,
                 residual: bool = True,
                 dropout: float = 0.,
                 alpha: float = 0.2,
                 predictor_hidden_feats: int = 128,
                 predictor_dropout: float = 0.,
                 number_atom_features: int = 30):

        super(GATModel, self).__init__()

        num_gnn_layers = len(graph_attention_layers)

        if activation is not None:
            activation = [activation] * num_gnn_layers
        
        self.model = dgllife.model.GATPredictor(
            in_feats=number_atom_features,
            hidden_feats=graph_attention_layers,
            num_heads=[n_attention_heads] * num_gnn_layers,
            feat_drops=[dropout] * num_gnn_layers,
            attn_drops=[dropout] * num_gnn_layers,
            alphas=[alpha] * num_gnn_layers,
            residuals=[residual] * num_gnn_layers,
            agg_modes=agg_modes,
            activations=activation,
            n_tasks=out_size,
            predictor_hidden_feats=predictor_hidden_feats,
            predictor_dropout=predictor_dropout)

    def forward(self, g):
        out = self.model(g, g.ndata['x'])

        return out