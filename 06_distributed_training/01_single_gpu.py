import torch
from torch.utils.data import DataLoader
import argparse
from utils import DemoDataset, Trainer


def main(config):
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_data = DataLoader(DemoDataset(2048),
                            batch_size=config.batch_size,
                            pin_memory=True,
                            shuffle=True
                            )
    trainer = Trainer(model, train_data, optimizer, config.device, config.save_every)
    trainer.train(config.total_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=3, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training on CPU is not supported"
    args.device = 0
    main(args)
