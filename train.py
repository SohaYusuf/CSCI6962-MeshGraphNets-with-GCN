import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np
import matplotlib.pyplot as plt

dataset_dir = "/home/yusufs/meshGraphNets_pytorch/dataset/data_cylinder/"
batch_size = 20
noise_std=2e-2

print_batch = 10
save_batch = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer= torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):
    train_loss_list = []

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))
            train_loss_list.append(loss.item())

        if batch_index % save_batch == 0:
            model.save_checkpoint()

        if batch_index == 100:
            validate(simulator, valid_loader)
        
        # if batch_index == 300:
        #     break
    
    # Plotting the training loss
    plt.figure(1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('True')
    plt.show()
    plt.savefig('/home/yusufs/meshGraphNets_pytorch/loss/train_loss_fig.png')
    np.savetxt('/home/yusufs/meshGraphNets_pytorch/loss/train_loss.txt', train_loss_list)

def validate(model:Simulator, dataloader):
    valid_loss_list = []

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.cuda()

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        print('batch %d [loss %.2e]'%(batch_index, loss.item()))
        valid_loss_list.append(loss.item())

        # if batch_index == 200:
        #     break

    # Plotting the training loss
    plt.figure(2)
    plt.plot(valid_loss_list, label='Validation Loss')
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('True')
    plt.savefig('/home/yusufs/meshGraphNets_pytorch/loss/valid_loss_fig.png', )
    plt.show()
    np.savetxt('/home/yusufs/meshGraphNets_pytorch/loss/valid_loss.txt', valid_loss_list)


if __name__ == '__main__':

    dataset_fpc = FPC(dataset_dir=dataset_dir, split='train', max_epochs=1)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])


    # validate
    valid_dataset_fpc = FPC(dataset_dir=dataset_dir, split='valid', max_epochs=1)
    valid_loader = DataLoader(dataset=valid_dataset_fpc, batch_size=1)

    # train
    train(simulator, train_loader, optimizer)