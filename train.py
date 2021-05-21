import torch.nn as nn
import torch

import torchvision
from torchvision.transforms import ToTensor
from autoencoder import AutoEncoder

c = {
    "epochs": 20,
    "lr": 0.01,
}


trainset = torchvision.datasets.MNIST("/media/sinclair/datasets",
                                      train=True, download=True, transform=ToTensor())

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

model = AutoEncoder(latent_dim=20, input_size=1)

optimizer = torch.optim.SGD(model.parameters(), lr=c["lr"])

loss = nn.MSELoss()

for e in range(c["epochs"]):
    for index, data in enumerate(trainloader):

        x, _  = data

        optimizer.zero_grad()

        xhat = model(x)
        reconstruction_loss = loss(xhat, x)

        reconstruction_loss.backward()
        optimizer.step()

        if index % 500 == 0:
            print(f"epoch: {e}, step: {index}, loss:{reconstruction_loss.item()}")





