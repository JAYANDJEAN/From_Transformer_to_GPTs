import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from utils import DemoDataset, Trainer


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class MultiTrainer(Trainer):
    def __init__(self, model: Module, train_data: DataLoader, optimizer: Optimizer, gpu_id: int, save_every: int):
        super().__init__(model, train_data, optimizer, gpu_id, save_every)
        self.model = DDP(model, device_ids=[gpu_id])


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)

    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_data = DataLoader(DemoDataset(2048),
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=False,
                            sampler=DistributedSampler(DemoDataset(2048))
                            )
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=3, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
