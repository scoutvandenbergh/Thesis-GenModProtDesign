import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import math
from rotary_embedding_torch import RotaryEmbedding

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
        return x.reshape(*self.args)


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


class ModelBackbone(pl.LightningModule):
    def __init__(self, learning_rate = 0.0001, n_warmup_steps = 1000):
        super().__init__()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):  
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < self.hparams.n_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.n_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate
    

    def n_params(self):
        params_per_layer = [(name, p.numel()) for name, p in self.named_parameters()]
        total_params = sum(p.numel() for p in self.parameters())
        params_per_layer += [('total', total_params)]
        return params_per_layer


class VAE(ModelBackbone):
    def __init__(
        self,
        vocab_size = 21, 
        hidden_sizes = [64, 128, 256, 512, 1024], 
        bottleneck_size = 128, 
        learning_rate = 0.0001, 
        blocks_per_stage = 4,
        n_heads = 8,
        n_warmup_steps = 1000,
        beta = 0.001):

        super().__init__(learning_rate = learning_rate, n_warmup_steps = n_warmup_steps)
        self.save_hyperparameters()
        

        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_sizes[0]), # gives B x 1024 x 128
            Permute(0,2,1), # B x 128 x 1024
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size = 4, stride = 4), # 1024 -> 256
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size = 4, stride = 4), # 256 -> 64
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size = 4, stride = 4), # 64 -> 16
            *[ResidualBlock(hidden_sizes[3], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.Conv1d(hidden_sizes[3], hidden_sizes[4], kernel_size = 4, stride = 4), # 16 -> 4
            *[ResidualBlock(hidden_sizes[4], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            View(-1, hidden_sizes[4]*4),
            nn.Linear(hidden_sizes[4]*4, bottleneck_size),
            nn.GELU()
        )
        
        self.lin_mu = nn.Linear(bottleneck_size, bottleneck_size)
        self.lin_var = nn.Linear(bottleneck_size, bottleneck_size)

        # note that I use quite mirrored encoder and decoder here
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_sizes[4]*4),
            nn.GELU(),
            View(-1, hidden_sizes[4], 4), # put them into B x 256 x 4, just as it was in last conv of encoder
            *[ResidualBlock(hidden_sizes[4], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.ConvTranspose1d(hidden_sizes[4], hidden_sizes[3], kernel_size = 4, stride = 4), # 4 -> 16
            *[ResidualBlock(hidden_sizes[3], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.ConvTranspose1d(hidden_sizes[3], hidden_sizes[2], kernel_size = 4, stride = 4), # 4 -> 16
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.ConvTranspose1d(hidden_sizes[2], hidden_sizes[1], kernel_size = 4, stride = 4), # 4 -> 16
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.ConvTranspose1d(hidden_sizes[1], hidden_sizes[0], kernel_size = 4, stride = 4), # 4 -> 16
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            nn.Conv1d(hidden_sizes[0], vocab_size, kernel_size = 1), # to output
        )

        # self.decoderL = nn.Sequential(
        #     nn.Linear(bottleneck_size, 64),
        #     nn.GELU(),
        #     nn.Linear(64,32),
        #     nn.GELU(),
        #     nn.Linear(32,16),
        #     nn.GELU(),
        #     nn.Linear(16,4),
        #     nn.GELU(),
        #     nn.Linear(4,1)
        # )

        self.beta = beta

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    # def decodeL(self, x):
    #     return self.decodeL(x) #added to predict length of protein without padding

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar #, self.decodeL(z) #added decodeL to forward pass
        
    def training_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('test_loss', loss)
        return loss

class VAE_transformer(ModelBackbone):
    def __init__(
        self,
        vocab_size = 21, 
        hidden_sizes = [64, 128, 256, 512, 1024], 
        bottleneck_size = 128, 
        learning_rate = 0.0001, 
        blocks_per_stage = 4,
        n_heads = 8,
        n_warmup_steps = 1000,
        beta = 0.001):

        super().__init__(learning_rate = learning_rate, n_warmup_steps = n_warmup_steps)
        self.save_hyperparameters()
        

        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_sizes[0]), # gives B x 1024 x 128
            Permute(0,2,1), # B x 128 x 1024
            #*[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[0], hidden_sizes[0], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size = 4, stride = 4), # 1024 -> 256
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[1], hidden_sizes[1], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size = 4, stride = 4), # 256 -> 64
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[2], hidden_sizes[2], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size = 4, stride = 4), # 64 -> 16
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[3], hidden_sizes[3], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.Conv1d(hidden_sizes[3], hidden_sizes[4], kernel_size = 4, stride = 4), # 16 -> 4
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[4], hidden_sizes[4], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            View(-1, hidden_sizes[4]*4),
            nn.Linear(hidden_sizes[4]*4, bottleneck_size),
            nn.GELU()
        )
        
        self.lin_mu = nn.Linear(bottleneck_size, bottleneck_size)
        self.lin_var = nn.Linear(bottleneck_size, bottleneck_size)

        # note that I use quite mirrored encoder and decoder here
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_sizes[4]*4),
            nn.GELU(),
            View(-1, hidden_sizes[4], 4), # put them into B x 256 x 4, just as it was in last conv of encoder
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[4], hidden_sizes[4], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.ConvTranspose1d(hidden_sizes[4], hidden_sizes[3], kernel_size = 4, stride = 4), # 4 -> 16
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[3], hidden_sizes[3], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.ConvTranspose1d(hidden_sizes[3], hidden_sizes[2], kernel_size = 4, stride = 4), # 4 -> 16
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[2], hidden_sizes[2], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.ConvTranspose1d(hidden_sizes[2], hidden_sizes[1], kernel_size = 4, stride = 4), # 4 -> 16
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[1], hidden_sizes[1], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.ConvTranspose1d(hidden_sizes[1], hidden_sizes[0], kernel_size = 4, stride = 4), # 4 -> 16
            Permute(0,2,1),
            *[TransformerRoPE(hidden_sizes[0], hidden_sizes[0], n_heads, dropout = 0.2) for _ in range(blocks_per_stage)],
            Permute(0,2,1),
            nn.Conv1d(hidden_sizes[0], vocab_size, kernel_size = 1), # to output
        )

        # self.decoderL = nn.Sequential(
        #     nn.Linear(bottleneck_size, 64),
        #     nn.GELU(),
        #     nn.Linear(64,32),
        #     nn.GELU(),
        #     nn.Linear(32,16),
        #     nn.GELU(),
        #     nn.Linear(16,4),
        #     nn.GELU(),
        #     nn.Linear(4,1)
        # )

        self.beta = beta

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    # def decodeL(self, x):
    #     return self.decodeL(x) #added to predict length of protein without padding

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar #, self.decodeL(z) #added decodeL to forward pass
        
    def training_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, _ = batch
        recon_x, mean, log_var = self.forward(X)
        reconstruction_loss = F.cross_entropy(recon_x, X)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence #implement loss for correct length
        self.log('test_loss', loss)
        return loss

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        self.kqv = nn.Linear(hidden_size, hidden_size * 3, bias=False) 
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        k, q, v = self.kqv(x).chunk(3, dim=-1)
        
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)

        attn = torch.matmul(q, k.transpose(-2,-1)) * x.size(-1)**0.5 #(B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        attention = F.softmax(attn, dim=-1)
        attention = torch.matmul(self.dropout(attention), v) #(B, nh, L, hs)
        
        attention = attention.transpose(1, 2).contiguous().view(B, L, C) # (B, L, nh, hs)
        attention = self.output(attention)
        print(attention.shape)
        
        return attention

class RoPEMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout): #add argument to choose for rotary or abs, or relative, or learned
        super().__init__()
        
        assert hidden_size % n_heads == 0
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        self.kqv = nn.Linear(hidden_size, hidden_size * 3, bias=False) 
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        
    def forward(self, x):
        B, L, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        k, q, v = self.kqv(x).chunk(3, dim=-1)
        
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, L, hs)
        
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        attn = torch.matmul(q, k.transpose(-2,-1)) * k.size(-1)**0.5 #(B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        attention = F.softmax(attn, dim=-1)
        attention = torch.matmul(self.dropout(attention), v) #(B, nh, L, hs)
        
        attention = attention.transpose(1, 2).contiguous().view(B, L, C) # (B, L, nh, hs)
        attention = self.output(attention)
        
        return attention
    
class TransformerRoPE(nn.Module):
    def __init__(self, input_dim, hidden_size, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.RoPE_MHA = RoPEMultiHeadAttentionLayer(hidden_size, n_heads, dropout)
        self.ffn = SwiGLU(input_dim)
        
    def forward(self, x):
        pre_ln = self.ln1(x)
        attn = self.RoPE_MHA(pre_ln)
        residual = attn + x

        pre_ln = self.ln2(residual)
        ffn = self.ffn(pre_ln)
        output = ffn + residual
        return output
    
class GLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        gates = self.linear(x).chunk(2, dim=-1) #split into 2 parts
        #sigmoid(xW + b) * (xV+c)
        activation = torch.sigmoid(gates[0]) * gates[1]
        return activation

class SwiGLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.beta = nn.Parameter(torch.ones(input_dim))
        #nn.Parameter tells pytorch to train this parameter

    def forward(self, x):
        gates = self.linear(x).chunk(2, dim=-1) #split into 2 parts
        #Swish_beta(xW+b) * (xV+c)
        activation = (gates[0] * torch.sigmoid(self.beta * gates[1])) * gates[1]
        return activation

class GeGLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)

    def forward(self, x):
        gates = self.linear(x).chunk(2, dim=-1) #split into 2 parts
        #GELU(xW+b)*(xV+c), GELU from https://arxiv.org/abs/1606.08415
        activation = (0.5 * gates[0] * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (gates[0] + 0.044715 * torch.pow(gates[0], 3.0))))) * gates[1] #approx
        #activation = gates[0] * 1/2 * (1 + torch.erf(gates[0] / (math.sqrt(2)))) * gates[1] #more exact?
        return activation