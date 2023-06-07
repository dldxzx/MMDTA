import numpy as np

# 定义氨基酸
Amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
# 定义缩写氨基酸
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
Amino_acids_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

Amino_acids_dict = {}
for i in range(len(Amino_acids)):
    Amino_acids_dict[Amino_acids[i]] = Amino_acids_num[i]

# 定义SMILES格式
atom_list = ['C', 'H', 'O', 'N', 'F', 'S', 'P', 'I', 'Cl', 'As', 'Se', 'Br', 'B', 'Pt', 'V', 'Fe', 'Hg', 'Rh', 'Mg', 'Be', 'Si', 'Ru', 'Sb', 'Cu', 'Re', 'Ir', 'Os']

Amino_acids_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

def getCompoundGraph(pdbid):
    with open('data/drug_sdf/' + pdbid + '.sdf', 'r') as read:
        read.readline()
        read.readline()
        read.readline()
        atom_num = 0
        edge_list = []
        node_list = []
        features = []
        while 1:
            line = read.readline()[:-1]
            if line == '$$$$':
                break
            line = line.split(' ')
            line = list(filter(None, line))
            if len(line) > 4 and line[3] in atom_list:
                atom_num += 1
                node_list.append(line[3])
            if len(line) == 6 and line[0].isdecimal() and line[-1].isdecimal():
                edge_list.append([int(line[0]) - 1, int(line[1]) - 1])
                edge_list.append([int(line[1]) - 1, int(line[0]) - 1])
    for atom in node_list:
        feature = [0 for _ in range(len(atom_list))]
        feature[atom_list.index(atom)] = 1
        features.append(feature)
    return atom_num, features, edge_list