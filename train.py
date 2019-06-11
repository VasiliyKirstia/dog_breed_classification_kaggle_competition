from typing import List

from torch import nn
import torch
import click


@click.option('--num-epochs', type=click.INT)
@click.option('--batch-size', type=click.INT)
@click.option('--num-workers', type=click.INT)
@click.option('--cuda-device-id', type=click.INT, multiple=True)
@click.option('--dataloader', type=click.Path(exists=True))
@click.option('--model', type=click.Path(exists=True))
def train(num_epochs: int, batch_size: int, num_workers: int, cuda_device_id: List[int], dataloader, model):
    print('Not ready yet :)')
    # for epoch in range(num_epochs):
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Finished Training')

    # todo: add training loop. Dataloader should be used from specified module (model should be loaded in the same manner)


if __name__ == '__main__':
    train()
