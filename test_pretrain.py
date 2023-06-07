import torch
from training import *
from models.MMDTA import Multimodal_Affinity

batch_size = 4
compound_sequence_dim = 65
protein_sequence_dim = 21
compound_test_data = CompoundDataset(root='data', dataset='test_data')
compound_test_loader = DataLoader(compound_test_data, batch_size=batch_size, shuffle=False)
model = Multimodal_Affinity(compound_sequence_dim, protein_sequence_dim, [32, 64, 128], 256).to(device)
model.load_state_dict(torch.load('data/pretrained_model/model_params.pt'))
test_labels, test_preds = validation(model, compound_test_loader)
test_result = [mae(test_labels, test_preds), rmse(test_labels, test_preds), pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds), r_squared(test_labels, test_preds)]
print(test_result)
