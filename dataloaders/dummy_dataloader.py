from torch.utils.data import DataLoader
from dataset import DogBreedDataset

__all__ = ['get_dataloader']


def get_dataloader(test, batch_size, num_workers):
    if test:
        return DataLoader(DogBreedDataset(is_test=False), batch_size=batch_size, num_workers=num_workers)
    else:
        return DataLoader(DogBreedDataset(is_test=False), batch_size=batch_size, num_workers=num_workers)
