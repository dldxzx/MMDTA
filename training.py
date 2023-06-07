import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from util import *
from util import _to_onehot
from models.MMDTA import Multimodal_Affinity
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluate_metrics import *
import prettytable as pt

import networkx as nx
import matplotlib.pyplot as plt

device = torch.device('cuda:0')

writer = SummaryWriter()


def training(model, train_loader, optimizer, epoch, epochs):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='red')
    training_loss = 0.0
    # for batch, data in enumerate(compound_train_loader):
    for batch, data in loop:
        compound_sequence = torch.from_numpy(np.array(_to_onehot(data.id, 150))).to(torch.float).to(device)
        protein_sequence = torch.from_numpy(np.array(_to_onehot(data.id, 1000))).to(torch.float).to(device)
        protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
        output = model(compound_sequence, data.to(device), protein_sequence, protein_img)
        loss = criterion(output, data.y.view(-1, 1).to(torch.float).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        loop.set_description(f'Training Epoch [{epoch} / {epochs}]')
        loop.set_postfix(loss=loss.item())
    writer.add_scalar('Training loss', training_loss, epoch)
    print('Training Epoch:[{} / {}], Mean Loss: {}'.format(epoch, epochs, training_loss / 12993))

def validation(model, loader, epoch=1, epochs=1):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), colour='blue')
        # for batch, data in enumerate(loader):
        for batch, data in loop:
            compound_sequence = torch.from_numpy(np.array(_to_onehot(data.id, 150))).to(torch.float).to(device)
            protein_sequence = torch.from_numpy(np.array(_to_onehot(data.id, 1000))).to(torch.float).to(device)
            protein_img = torch.from_numpy(img_resize(data.id)).unsqueeze(1).to(torch.float).to(device)
            output = model(compound_sequence, data.to(device), protein_sequence, protein_img)
            loop.set_description(f'Testing Epoch [{epoch} / {epochs}]')
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    return total_labels, total_preds

def graph_showing(data):
    G = nx.Graph()
    edge_index = data.edge_index.t().numpy()
    G.add_edges_from(edge_index)
    nx.draw(G)
    plt.savefig('aa.png')

if __name__ == '__main__':
    # load dataset
    batch_size = 16
    train_data = CompoundDataset(root='data', dataset='train_data')
    val_data = CompoundDataset(root='data', dataset='val_data')
    test_data = CompoundDataset(root='data', dataset='test_data')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    epochs = 200
    epoch = 1
    compound_sequence_dim = 65
    protein_sequence_dim = 21
    
    model = Multimodal_Affinity(compound_sequence_dim, protein_sequence_dim, [32, 64, 128] , 256).to(device)
    # print(model)
    learning_rate = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.MSELoss().to(device)
    best_rmse = 1000

    for epoch in range(1, epochs + 1):
        training(model, train_loader, optimizer, epoch, epochs)
        # scheduler.step()
        val_labels, val_preds = validation(model, val_loader,epoch, epochs)
        test_labels, test_preds = validation(model, test_loader, epoch, epochs)
        val_result = [mae(val_labels, val_preds), rmse(val_labels, val_preds), pearson(val_labels, val_preds), spearman(val_labels, val_preds), r_squared(val_labels, val_preds)]
        test_result = [mae(test_labels, test_preds), rmse(test_labels, test_preds), pearson(test_labels, test_preds), spearman(test_labels, test_preds), ci(test_labels, test_preds), r_squared(test_labels, test_preds)]
        tb = pt.PrettyTable()
        tb.field_names = ['Epoch / Epochs', 'Set', 'MAE', 'RMSE', 'Pearson', 'Spearman', 'CI', 'R-Squared']
        tb.add_row(['{} / {}'.format(epoch, epochs), 'Validation', val_result[0], val_result[1], val_result[2], val_result[3], val_result[-1]])
        tb.add_row(['{} / {}'.format(epoch, epochs).format(epoch, epochs), 'Test', test_result[0], test_result[1], test_result[2], test_result[3], test_result[4], test_result[-1]])
        print(tb)
        writer.add_scalar('RMSE/Val RMSE', val_result[1], epoch)
        writer.add_scalar('RMSE/Test RMSE', test_result[1], epoch)
        with open('result/mmdta.txt', 'a') as write:
            write.writelines(str(tb) + '\n')
        if float(test_result[1]) < best_rmse:
            best_rmse = float(test_result[1])
            torch.save(model, 'data/best_model/mmdta.pt')
            torch.save(model.state_dict(), 'data/best_model/mmdta_params.pt')
