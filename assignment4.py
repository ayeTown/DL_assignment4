import os
import time
import threading
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self,num_classes = 10):
        super(NeuralNetwork, self).__init__()
        self.num_classes = num_classes
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1,10,kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 5,stride = 5),
            nn.Conv2d(10,10,kernel_size = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Flatten(),
            nn.Linear(40,num_classes),
            nn.ReLU()
        )
    def forward(self, x):
        logits = self.conv_stack(x)
        return logits


def train(num_epochs,batch_size,learning_rate):
    training_data = datasets.MNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor(),
    )
    test_data = datasets.MNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor(),
    )
    sampler = DistributedSampler(training_data)
    training_dataloader = DataLoader(training_data,batch_size = batch_size,sampler = sampler)
    model = NeuralNetwork(num_classes = 10)
    model = DistributedDataParallel(model)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    learning_rate *= world_size
    optimizer = optim.SGD(model.parameters(),lr = learning_rate)
    criterion = nn.CrossEntropyLoss()
    iter_losses = []
    epoch_losses = []
    elapsed_times = []
    epbar = range(num_epochs)
    start = perf_counter()
    for ep in epbar:
        ep_loss = 0.0
        num_iters = 0
        for X, Y in training_dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            iter_losses.append(loss.item())
            ep_loss += loss.item()
            loss.backward()
            optimizer.step()
            num_iters += 1
        ep_loss /= num_iters
        if rank == 0:
            print("epoch", ep, "num_iters", num_iters, "loss", ep_loss, "elapsed time (s)",perf_counter() - start)
        epoch_losses.append(ep_loss)
        elapsed_times.append(perf_counter()  - start)
        metrics = pd.DataFrame({'epoch_losses': epoch_losses, 'elapsed_time':
        elapsed_times})
        metrics.to_csv('metrics.csv')
    return iter_losses, epoch_losses


if __name__ ==  '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    import mpi4py.MPI as MPI
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print("World size:", world_size)
    dist.init_process_group('gloo',
        init_method = 'env://',
        world_size = world_size,
        rank = rank,
    )
    train()
