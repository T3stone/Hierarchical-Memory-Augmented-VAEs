import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, num_features,hidden_dims,latent_dim):
        super(Encoder,self).__init__()
        self.linears=nn.Sequential(
            nn.Linear(num_features,hidden_dims*2),
            nn.BatchNorm1d(hidden_dims*2),
            nn.ReLU(),
            nn.Linear(hidden_dims*2, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU())
        self.latent=nn.Linear(hidden_dims,latent_dim)
        self.get_mu=nn.Sequential(nn.Linear(hidden_dims*2, hidden_dims),
                                  nn.BatchNorm1d(hidden_dims),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dims, latent_dim))
        self.get_logvar=nn.Sequential(nn.Linear(hidden_dims*2, hidden_dims),
                                  nn.BatchNorm1d(hidden_dims),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dims, latent_dim))
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, image):
       
        x=self.linears[0](image)
        x=self.linears[1](x)
        x=self.linears[2](x)
        temp=x
        x=self.linears[3](x)
        x=self.linears[4](x)
        x=self.linears[5](x)
        x=self.latent(x)
        mu=self.get_mu(temp)
        logvar=self.get_logvar(temp)
        z=self._reparameterize(mu, logvar)
        return z,mu, logvar
 
class Decoder(nn.Module):
    def __init__(self,latent_dim,hidden_dim,image_channels,image_size):
        super(Decoder,self).__init__()

        self.linear1=nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.linear2=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
        )
        self.final_linear=nn.Linear(hidden_dim*2,image_size*image_size*image_channels)
    def forward(self, x): 
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.final_linear(x)                              
        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dim,hidden_dim,image_channels,image_size):
        super(VAE,self).__init__()
        self.encoder=Encoder(num_features=image_size*image_size*image_channels, hidden_dims=hidden_dim, latent_dim=latent_dim)
        self.decoder=Decoder(latent_dim,hidden_dim,image_channels,image_size)
    def forward_loss(self,image):
        image = image.view(image.size(0), -1) 
        z,mu, logvar=self.encoder(image)
        x_reconstruct=self.decoder(z)
        loss=0
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss1=nn.functional.mse_loss(x_reconstruct,image,reduction='sum')
        loss=kl_loss+mse_loss1
        return loss
    def forward(self, image):
        image = image.view(image.size(0), -1) 
        z,mu, logvar=self.encoder(image)
        x_reconstruct=self.decoder(z)
        loss=self.forward_loss(image)
        return x_reconstruct,loss