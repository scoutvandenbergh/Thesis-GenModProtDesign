import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class Permute(nn.Module): 
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)

class View(nn.Module): # same thing as permute but then for View
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view(*self.args)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim = 64, kernel_size = 5, dropout = 0.2): # a simple residual block
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding = "same"), #channels in, channels out, kernel size
            nn.GELU(),
            nn.Dropout(dropout),
            Permute(0,2,1),
            nn.LayerNorm(hidden_dim),
            Permute(0,2,1) 
            )
        # NOTE: these permutes are necessary here because LayerNorm expects the "hidden_dim" to be the last dim
        # While for Conv1d the "channels" or "hidden_dims" are the second dimension.
        # These permutes basically swap BxCxL to BxLxC for the layernorm, and afterwards swap them back

    def forward(self, x):
        return self.net(x) + x #residual connection


class VerySimpleMLPVAE(pl.LightningModule):
    def __init__(self, learning_rate = 0.0001):
        """
        This VAE doesn't even use a convolution
        just an embedding layer, puts those embeddings as a 1D vector
        then one linear layer + GELU, then to bottleneck, then to decoder with also only 3 layers
        """
        super().__init__()
        self.learning_rate = learning_rate
        
        self.encoder = nn.Sequential(
            nn.Embedding(21, 8), # gives B x 1024 x 8
            View(-1, 1024*8), # gives B x 1024 x 8
            nn.Linear(1024*8, 128),
            nn.GELU()
        )
        
        self.lin_mu = nn.Linear(128, 128)
        self.lin_var = nn.Linear(128, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1024*8),
            nn.GELU(),
            View(-1, 1024, 8),
            nn.Linear(8, 21)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def VAE_loss(x, output, mean, logvar, beta = 0.001):
        reconstruction_loss = F.cross_entropy(output.permute(0,2,1), x)
        KL_divergence = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        return reconstruction_loss + beta * KL_divergence
        
    def training_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer

print(VerySimpleMLPVAE().n_params()[-1]) #2155365 params (2.16M)

class CrudeConvVAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0001):
        """
        This VAE uses one conv and crudely maxpools them across all positions.
        Conceptually, the encoder can learn 512 local patterns
        and after max pooling, it doesn't matter where it found them.

        The decoder needs a specific strategy here to get back to 1024 positions
        """
        super().__init__()
        self.learning_rate = learning_rate
        
        self.encoder = nn.Sequential(
            nn.Embedding(21, 128), # gives B x 1024 x 8
            Permute(0,2,1),
            nn.Conv1d(128, 128, 15, padding = "same"),
            nn.GELU(),
            nn.MaxPool1d(1024),
            View(-1, 128*1),
            nn.Linear(128, 128),
            nn.GELU()
        )
        
        self.lin_mu = nn.Linear(128, 128)
        self.lin_var = nn.Linear(128, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.GELU(),
            View(-1, 1, 1024), # act as if the 1024 neurons are now positions, every position has a 1D representation
            nn.Conv1d(1, 64, 15, padding = "same"), # just a conv sprinkled in
            nn.GELU(),
            nn.Conv1d(64, 21, 1), # to output
            Permute(0,2,1)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def VAE_loss(x, output, mean, logvar, beta = 0.001):
        reconstruction_loss = F.cross_entropy(output.permute(0,2,1), x)
        KL_divergence = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        return reconstruction_loss + beta * KL_divergence
        
    def training_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        loss = self.VAE_loss(X, recon_x, mean, log_var)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer

print(CrudeConvVAE().n_params()[-1]) #432597 params (0.433M)

class VAE(pl.LightningModule):
    def __init__(self, learning_rate = 0.0001):
        """
        This would be my prefered VAE implementation
        Instead of max poolings it uses strided convs
        A strided conv where stride = kernel size, functions the same way max pooling does,
        but then learns filters for the pooling instead of max

        Also for the decoder there is convtranspose1d instead of upsample.

        Take not of how deep this model is with how few parameters. (1.3 mill)
        This is the power of aggresive pooling (through strided convs)
        """
        super().__init__()
        self.learning_rate = learning_rate
        
        self.encoder = nn.Sequential(
            nn.Embedding(21, 128), # gives B x 1024 x 128
            Permute(0,2,1), # B x 128 x 1024
            ResidualBlock(128, 5, 0.2),
            ResidualBlock(128, 5, 0.2),
            ResidualBlock(128, 5, 0.2),
            nn.Conv1d(128, 256, 4, stride = 4), # 1024 -> 256: B x 256 x 256
            ResidualBlock(256, 5, 0.2),
            ResidualBlock(256, 5, 0.2),
            ResidualBlock(256, 5, 0.2),
            nn.Conv1d(256, 512, 4, stride = 4), # 256 -> 64: B x 512 x 64
            ResidualBlock(512, 5, 0.2),
            ResidualBlock(512, 5, 0.2),
            ResidualBlock(512, 5, 0.2),
            nn.Conv1d(512, 1024, 4, stride = 4), # 64 -> 16: B x 1024 x 16
            ResidualBlock(1024, 5, 0.2),
            ResidualBlock(1024, 5, 0.2),
            ResidualBlock(1024, 5, 0.2),
            nn.Conv1d(1024, 2048, 4, stride = 4), # 16 -> 4: B x 2048 x 4
            View(-1, 2048*4),
            nn.Linear(2048*4, 128),
            nn.GELU()
        )
        
        self.lin_mu = nn.Linear(128, 128)
        self.lin_var = nn.Linear(128, 128)

        # note that I use quite mirrored encoder and decoder here
        self.decoder = nn.Sequential(
            nn.Linear(128, 2048*4),
            nn.GELU(),
            View(-1, 2048, 4), # put them into B x 256 x 4, just as it was in last conv of encoder
            nn.ConvTranspose1d(2048, 1024, 4, stride = 4), # 4 -> 16
            ResidualBlock(1024, 5, 0.2),
            ResidualBlock(1024, 5, 0.2),
            ResidualBlock(1024, 5, 0.2),
            nn.ConvTranspose1d(1024, 512, 4, stride = 4), # 16 -> 64
            ResidualBlock(512, 5, 0.2),
            ResidualBlock(512, 5, 0.2),
            ResidualBlock(512, 5, 0.2),
            nn.ConvTranspose1d(512, 256, 4, stride = 4), # 64 -> 256
            ResidualBlock(256, 5, 0.2),
            ResidualBlock(256, 5, 0.2),
            ResidualBlock(256, 5, 0.2),
            nn.ConvTranspose1d(256, 128, 4, stride = 4), # 256 -> 1024
            ResidualBlock(128, 5, 0.2),
            ResidualBlock(128, 5, 0.2),
            ResidualBlock(128, 5, 0.2),
            nn.Conv1d(128, 21, 1), # to output
            Permute(0,2,1)
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate) #also look at LR schedulers
        return optimizer
    
    #def VAE_loss(x, output, mean, logvar, beta = 0.001): couldnt get it to work using this and then 
    # referring to it as self.VAE_loss(...) in evaluation_step

        #print(output, output.shape)
        #reconstruction_loss = F.cross_entropy(output, x) #was output.permute(0,2,1)
        #KL_divergence = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        #return reconstruction_loss + beta * KL_divergence
        
    def training_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1), X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + 0.001 * KL_divergence
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): #need batch_idx? otherwise errors
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1), X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + 0.001 * KL_divergence
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1), X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + 0.001 * KL_divergence
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer

print(VAE().n_params()[-1]) #66245653 params (66M)