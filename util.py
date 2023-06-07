import h5py, math, os, torch
import pandas as pd
import numpy as np
import cv2
from Bio import SeqIO
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA

smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

amino_acids = ['PAD','A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

atom_list = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'I', 'Cl', 'As', 'Se', 'Br', 'B', 'Pt', 'V', 'Fe', 'Hg', 'Rh', 'Mg', 'Be', 'Si', 'Ru', 'Sb', 'Cu', 'Re', 'Ir', 'Os']



def smiles2onehot(pdbid):
    seq = pd.read_csv('data/drug_smiles/' + pdbid + '.smi', header=None).to_numpy().tolist()[0][0].split('\t')[0]
    integer_encoder = []
    onehot_encoder = []
    for item in seq:
        integer_encoder.append(smiles_dict[item])
    for index in integer_encoder:
        temp = [0 for _ in range(len(smiles_dict) + 1)]
        temp[index] = 1
        onehot_encoder.append(temp)
    return onehot_encoder

# print(smiles2onehot('5a7b'))

def protein2onehot(pdbid):
    for seq_recoder in SeqIO.parse('data/target_fasta/' + pdbid + '.fasta', 'fasta'):
        seq = seq_recoder.seq
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


def _to_onehot(data, max_len):
    feature_list = []
    for seq in data:
        if max_len == 1000:
            feature = protein2onehot(seq)
            if len(feature) > 1000:
                feature = feature[:1000]
            feature_list.append(feature)
        elif max_len == 150:
            feature = smiles2onehot(seq)
            if len(feature) > 150:
                feature = feature[:150]
            feature_list.append(feature)
        else:
            print('max length error!')
    for i in range(len(feature_list)):
        if len(feature_list[i]) != max_len:
            for j in range(max_len - len(feature_list[i])):
                if max_len == 1000:
                    temp = [0] * 21
                    temp[0] = 1
                elif max_len == 150:
                    temp = [0] * 65
                    temp[0] = 1
                feature_list[i].append(temp)
    return torch.from_numpy(np.array(feature_list, dtype=np.float32))



def img_resize(data):
    data_list = []
    for id in data:
        img = np.load('data/distance_matrix/' + id + '.npz')['map']
        img_resize = cv2.resize(img, [224, 224], interpolation = cv2.INTER_AREA)
        data_list.append(img_resize)
    return np.array(data_list)

class CompoundDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None , compound=None, protein=None, affinity=None, transform=None, pre_transform=None, compound_graph=None, protein_graph=None):
        super(CompoundDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(compound, affinity, compound_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self, compound, affinity, compound_graph):
        assert (len(compound) == len(affinity)), '这两个列表必须是相同的长度!'
        data_list = []
        data_len = len(compound)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len))
            smiles = compound[i]
            # target = protein[i]
            label = affinity[i]
            print(smiles)
            # print(target)
            print(label)

            size, features, edge_index = compound_graph[i][smiles]
            GCNCompound = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.FloatTensor([label]), id=smiles)
            GCNCompound.__setitem__('size', torch.LongTensor([size]))
            data_list.append(GCNCompound)
            # data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
