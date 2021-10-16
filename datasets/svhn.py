"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

import params


def get_svhn(train):
    """Get SVHN dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                                      transforms.Resize((224,224))])
    # dataset and data loader
    svhn_dataset = datasets.SVHN(root=params.data_root,
                                   #train=train,
                                   transform=pre_process,
                                   download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return svhn_data_loader
