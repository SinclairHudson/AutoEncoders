import torch.nn as nn
from autoencoder import Decoder  # decoder will be the same


class VAEDecoder(nn.Module):


class VariationalAutoEncoder(nn.Module):
    """
    Variational Auto Encoder, going down into the latent space.
    The forward pass produces two tensors; one represents the mean for the probability
    distributions of all the dimensions in the latent space, the other for the variance.

    """
    def __init__(self, latent_dim, input_size):
        """
        :param latent_dim: dimensionality of the latent space.
        More dimensions = less compression
        :param input_size: the number of channels in the input image

        """
        super(VAEDecoder, self).__init__()
        self.enc = VAEEncoder(latent_dim, input_size)
        self.dec = Decoder(latent_dim, input_size)

    def forward(self, x):
        mean, variance = self.enc(x)

        variance = torch.exp(0.5 * variance)


        xhat = self.dec(z)
        return xhat
