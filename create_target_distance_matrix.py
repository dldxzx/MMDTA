import math
from Bio.PDB.PDBParser import PDBParser

parser = PDBParser(PERMISSIVE=1)
# 生成距离矩阵并进行存储
def generate_dis_metirc(pdbid):
    CA_Metric = []
    coordinate_list = []
    structure = parser.get_structure(pdbid, 'data/target_pdb/' + pdbid + '.pdb')
    for chains in structure:
        for chain in chains:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == 'CA':
                        coordinate_list.append(list(atom.get_vector()))
    for i in range(len(coordinate_list)):
        ca_raw_list = []
        for j in range(len(coordinate_list)):
            if i == j:
                ca_raw_list.append(0)
            else:
                ca_raw_list.append(math.sqrt((coordinate_list[i][0]- coordinate_list[j][0]) ** 2 + (coordinate_list[i][1] - coordinate_list[j][1]) ** 2 + (coordinate_list[i][2] - coordinate_list[j][2]) ** 2))
        CA_Metric.append(ca_raw_list)
    return CA_Metric

print(generate_dis_metirc('1a30'))