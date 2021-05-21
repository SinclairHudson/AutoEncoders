import torch.nn as nn
import torch

class Encoder(nn.Module):
    """
    Autoencoder encoder, going down into the latent space
    """
    def __init__(self, latent_dim, input_size):
        """
        :param latent_dim: dimensionality of the latent space.
        More dimensions = less compression
        :param input_size: the number of channels in the input,
        which for the decoder is the number of channels to output
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=3,
                               kernel_size=(3,3), stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=(3,3), stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=latent_dim,
                               kernel_size=(7,7))
        self.a = nn.LeakyReLU()


    def forward(self, x):
        x = self.a(self.conv1(x))
        x = self.a(self.conv2(x))
        x = self.a(self.conv3(x))
        # z is a long boy, batch_size x latent_dim x 1 x 1
        return x


class Decoder(nn.Module):
    """
    Autoencoder decoder, constructing the input from the latent representation
    """
    def __init__(self, latent_dim, output_size):
        """
        :param latent_dim: dimensionality of the latent space.
        More dimensions = less compression
        :param input_size: the number of channels in the input,
        which for the decoder is the number of channels to output
        """
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=latent_dim, out_channels=4,
                                        kernel_size=(3,3), stride=2)
        self.a = nn.LeakyReLU()
        self.conv2 = nn.ConvTranspose2d(in_channels=4, out_channels=4,
                                        kernel_size=(3,3), stride=3)
        self.conv3 = nn.ConvTranspose2d(in_channels=4, out_channels=1,
                                        kernel_size=(4,4), stride=3)


    def forward(self, z):
        # z is a long boy, batch_size x latent_dim x 1 x 1
        x = self.a(self.conv1(z))
        x = self.a(self.conv2(x))
        x = self.a(self.conv3(x))
        return x

class AutoEncoder(nn.Module):
    """
    Autoencoder encoder, going down into the latent space
    """
    def __init__(self, latent_dim, input_size):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(latent_dim, input_size)
        self.dec = Decoder(latent_dim, input_size)

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat

