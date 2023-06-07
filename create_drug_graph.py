import os
import numpy as np
import pandas as pd
# from rdkit.Chem import AllChem

from graphUtil import getCompoundGraph
from util import *

train_id = pd.read_csv('data/train_data.csv')['PDBID'].to_numpy().tolist()
val_id = pd.read_csv('data/val_data.csv')['PDBID'].to_numpy().tolist()
test_id = pd.read_csv('data/test_data.csv')['PDBID'].to_numpy().tolist()

train_affinity = pd.read_csv('data/train_data.csv')['affinity'].to_numpy().tolist()
val_affinity = pd.read_csv('data/val_data.csv')['affinity'].to_numpy().tolist()
test_affinity = pd.read_csv('data/test_data.csv')['affinity'].to_numpy().tolist()



train_drug_graph = []
test_drug_graph = []

for i in range(len(test_id)):
    drug_info = {}
    g = getCompoundGraph(test_id[i])
    drug_info[test_id[i]] = g
    test_drug_graph.append(drug_info)
print('测试集药物转换完成')

test_id, test_affinity =  np.asarray(test_id), np.asarray(test_affinity)

print('准备将药物测试集数据转化为Pytorch格式')
protein_train_data = CompoundDataset(root='data', dataset='test_data',
                                    compound=test_id, compound_graph=test_drug_graph, affinity=test_affinity)

print('药物测试集集数据转化为Pytorch格式完成')

