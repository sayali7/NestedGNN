#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
"""
Created By  : Sayali Anil Alatkar 
Created Date: 04/29/2023 
version ='1.0'
"""
# ---------------------------------------------------------------------------
# Implementation for NestedGNN:Detecting Malicious Network Activity with Nested Graph Neural Networks 
# (doi: 10.1109/ICC45855.2022.9838698.)

import os
import random
import pickle
from collections import OrderedDict, Counter
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

import dgl
import dgl.data
from dgl.nn import GraphConv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, average_precision_score

from captum.attr import IntegratedGradients
from functools import partial

import pandas as pd
import numpy as np


class NestedGNN(nn.Module):
    def __init__(self, in_feats, out_feats, h_feats, num_classes):
        super(NestedGNN, self).__init__()
        # Inner GNN Layers
        self.conv1 = GraphConv(in_feats, h_feats[0], allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats[0], h_feats[1], allow_zero_in_degree=True)
        # Outer GNN Layers
        self.conv3 = GraphConv(out_feats + h_feats[1], h_feats[2], allow_zero_in_degree=True)
        self.conv4 = GraphConv(h_feats[2], h_feats[3], allow_zero_in_degree=True)
        self.conv5 = GraphConv(h_feats[3], h_feats[4], allow_zero_in_degree=True)
        self.conv6 = GraphConv(h_feats[4], h_feats[5], allow_zero_in_degree=True)
        #self.conv7 = GraphConv(h_feats[5], h_feats[6], allow_zero_in_degree=True)
        # Classification Layer (downstream task)
        self.classify = nn.Linear(h_feats[5], num_classes)

    def forward(self, g, g_out, in_layer_feat, out_layer_feat, outer_edge_weight=None, inner_edge_weight=None): 
        
        # Generate node embeddings for inner GNN
        h = self.conv1(g, in_layer_feat, edge_weight = inner_edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight = inner_edge_weight)
        h = F.relu(h)
        g.ndata["h"] = h
        
        # Reduce (similar to readout for inner GNN)
        reduced_in_layer = dgl.mean_nodes(g, "h") # average (can be replaced with sum,max,..)
        #print (reduced_in_layer.shape)
        # Merge 
        merged_out_layer_feat = torch.cat((out_layer_feat,reduced_in_layer),-1)
        
        # Generate node emb for outer GNN
        h = self.conv3(g_out, merged_out_layer_feat, edge_weight = outer_edge_weight)
        h = F.relu(h)
        h = self.conv4(g_out, h, edge_weight = outer_edge_weight)
        h = F.relu(h)
        h = self.conv5(g_out, h, edge_weight = outer_edge_weight)
        h = F.relu(h)
        h = self.conv6(g_out, h, edge_weight = outer_edge_weight)
        h = F.relu(h)
        #h = self.conv7(g_out, h)
        #h = F.relu(h)
        g_out.ndata["h"] = h
        
        # readout
        hg = dgl.mean_nodes(g_out, 'h')
        return self.classify(hg)
