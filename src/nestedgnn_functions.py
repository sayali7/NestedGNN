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

#from sklearn.model_selection import train_test_split, KFold
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import balanced_accuracy_score, confusion_matrix, average_precision_score, precision_recall_curve, f1_score, auc

import pandas as pd
import numpy as np

# SPLIT TRAIN/TEST
def generate_dataset(new_PsychAD2_GRN_Dataset,PsychAD2_5_CC_Dataset, patient_ids,
                     train_labels,test_labels,train_patient_ids, test_patient_ids,
                    num_cell_types = 12, OUTER_LAYER_BATCH = 10, INNER_LAYER_BATCH = 10*12):
    
    class GRN_train(DGLDataset):
        def __init__(self):
            super().__init__(name="GRN")

        def process(self):
            self.graphs = train_grn_graphs
            self.labels = train_grn_labels

        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)
    
    class GRN_test(DGLDataset):
        def __init__(self):
            super().__init__(name="GRN")

        def process(self):
            self.graphs = test_grn_graphs
            self.labels = test_grn_labels


        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)

    class CC_train(DGLDataset):
        def __init__(self):
            super().__init__(name="CC")

        def process(self):
            self.graphs = train_cc_graphs
            self.labels = train_labels

        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)

    class CC_test(DGLDataset):
        def __init__(self):
            super().__init__(name="CC")

        def process(self):
            self.graphs = test_cc_graphs
            self.labels = test_labels


        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)

    #train_patient_ids, test_patient_ids, train_labels, test_labels = train_test_split(patient_ids,labels, test_size=0.1, random_state=42, stratify=labels)

    train_grn_labels = [val for val in train_labels for _ in range(12)]
    test_grn_labels = [val for val in test_labels for _ in range(12)]

    train_dataset_grn, test_dataset_grn={},{}
    train_dataset_cc, test_dataset_cc={},{}

    for pid in patient_ids:
        if pid in train_patient_ids:
            train_dataset_grn[pid] = new_PsychAD2_GRN_Dataset[pid]
            train_dataset_cc[pid] = PsychAD2_5_CC_Dataset[pid]
        else:
            test_dataset_grn[pid] = new_PsychAD2_GRN_Dataset[pid]
            test_dataset_cc[pid] = PsychAD2_5_CC_Dataset[pid]
            
     
    train_grn_graphs = [v1 for k,v in train_dataset_grn.items() for k1,v1 in v.items()]
    test_grn_graphs = [v1 for k,v in test_dataset_grn.items() for k1,v1 in v.items()]

    train_cc_graphs = [v for k,v in train_dataset_cc.items() ]
    test_cc_graphs = [v for k,v in test_dataset_cc.items()]

    train_grn_obj = GRN_train()
    test_grn_obj = GRN_test()

    train_cc_obj = CC_train()
    test_cc_obj = CC_test()
    
    # CREATE DATALOADERS
    train_grn_dataloader = GraphDataLoader(train_grn_obj, batch_size=INNER_LAYER_BATCH, drop_last=False)
    test_grn_dataloader = GraphDataLoader(test_grn_obj, batch_size=INNER_LAYER_BATCH, drop_last=False)

    train_cc_dataloader = GraphDataLoader(train_cc_obj, batch_size=OUTER_LAYER_BATCH, drop_last=False)
    test_cc_dataloader = GraphDataLoader(test_cc_obj, batch_size=OUTER_LAYER_BATCH, drop_last=False)

    return train_grn_dataloader,test_grn_dataloader,train_cc_dataloader,test_cc_dataloader


# SPLIT TRAIN/TEST
def generate_test_dataset(grn_dataset,cc_dataset, patient_ids,
                     test_labels,num_cell_types = 12, OUTER_LAYER_BATCH = 10, INNER_LAYER_BATCH = 10*12):

    class GRN_test(DGLDataset):
        def __init__(self):
            super().__init__(name="GRN_test")

        def process(self):
            self.graphs = test_grn_graphs
            self.labels = test_grn_labels

        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)

    class CC_test(DGLDataset):
        def __init__(self):
            super().__init__(name="CC_test")

        def process(self):
            self.graphs = test_cc_graphs
            self.labels = test_labels


        def __getitem__(self, i):
            return self.graphs[i], self.labels[i]

        def __len__(self):
            return len(self.graphs)

    #train_patient_ids, test_patient_ids, train_labels, test_labels = train_test_split(patient_ids,labels, test_size=0.1, random_state=42, stratify=labels)

    test_grn_labels = [val for val in test_labels for _ in range(12)]

    test_dataset_grn,test_dataset_cc={},{}

    for pid in patient_ids:
        test_dataset_grn[pid] = grn_dataset[pid]
        test_dataset_cc[pid] = cc_dataset[pid]
            
     
    test_grn_graphs = [v1 for k,v in test_dataset_grn.items() for k1,v1 in v.items()]
    test_cc_graphs = [v for k,v in test_dataset_cc.items()]

    test_grn_obj = GRN_test()
    test_cc_obj = CC_test()
    
    # CREATE DATALOADERS
    test_grn_dataloader = GraphDataLoader(test_grn_obj, batch_size=INNER_LAYER_BATCH, drop_last=False)
    test_cc_dataloader = GraphDataLoader(test_cc_obj, batch_size=OUTER_LAYER_BATCH, drop_last=False)

    return test_grn_dataloader,test_cc_dataloader