import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(BasicEncoder, self).__init__()

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latent vectors mu and logvar
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

         #############################################################################
        # TODO:                                                                     #
        #    1. linear -> relu to get hidden represntation                           #
        #    2. from hidden representation we have two heads.                       #
        #     one to produce mu and another to produce logvar. 
        #     both use the same latent dim.                                         #
        #############################################################################

        # simple 1 hidden layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # hidden layer
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # output mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # output logvar
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):   
        """ forward pass for vae encoder.
            args: 
                x: [N, input_dim]

            outputs:
                mu: [N, latent_dim]
                logvar: [N, latent_dim]
        """
        
        mu, logvar = None, None

         #############################################################################
        # TODO:                                                                     #
        #    1. implement forward pass for encoder.                        #
        #    2. return mu and logvar                                    #
        #############################################################################
        h = F.relu(self.fc1(x))   # hidden representation
        mu = self.fc_mu(h)        # mu
        logvar = self.fc_logvar(h) # logvar
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=256):

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latents

            students implement Basic VAE using encoder and decoder. 
            1. instantiate encoder and pass in variables.
            2. instantiate decoder and pass in variables.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim


        #############################################################################
        # TODO:                                                                     #
        #    1. instantiate encoder (from above) and decoder(imported) for vae.
        #   NOTE: replace the None with instantiation. 
        #   NOTE: Decoder implementation is already provided. 
        #       Only need to instantiate it with appropriate args                  #
        #############################################################################
        # instantiate encoder and decoder
        self.encoder = BasicEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = BasicDecoder(output_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def reparameterize(self, mu, logvar):
        """
            args:
                mu: [N,latent_dim]
                logvar: [N, latent_dim]
            outputs:
                z: reparameterized representation [N, latent_dim]

        """

        z = None

        #############################################################################
        # TODO:                                                                     #
        #    1. compute std from log-variance. 
        #    2. sample epsilon
        #    3. compute reparameterization.                                         #
        #############################################################################
        std = torch.exp(0.5 * logvar)          # σ = sqrt(var)
        eps = torch.randn_like(std)            # sample ε ~ N(0,1)
        z = mu + eps * std                     # reparameterization trick: z∼N(μ,σ2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return z
    
    @torch.jit.export
    def encode(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation [N, latent_dim]
                mu: mean latent [N, latent_dim]
                logvar: log variance latent [N, latent_dim]
        """
        z, mu, logvar = None, None, None

        #############################################################################
        # TODO:                                                                     #
        #    1. encode imput image (NOTE recall the vae takes a flattened input)
        #    2. compute reparameterization                                  #
        #############################################################################

        # flatten input
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return (z, mu, logvar)
    
    def forward(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation
                mu: mean latent
                logvar: log variance latent
        """
        out, mu, logvar = None, None, None
        #############################################################################
        # TODO:                                                                     #
        #    1. invoke encoder and decoder to obtain reconstructed output, mu and logvar #
        #############################################################################

        # if its flattened
        if len(x.shape) == 2:  # [N, 784]
            x_flat = x
        else:
            # flatten input
            x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x_flat)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return (out, mu, logvar)
        
    @torch.jit.export
    @torch.no_grad()
    def generate(self, z):
        """
            args:
                x: input [N, latent_dim]
            outputs:
                out: (N, input_dim)
        """
        out = None
        out = self.decoder(z)
        return out
