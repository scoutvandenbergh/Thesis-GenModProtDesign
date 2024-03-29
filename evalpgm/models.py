import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from evalpgm.blocks import Permute, View, ResidualBlock, ModelBackbone, TransformerRoPE, newGELU
from evalpgm.scripts.generate import generate_sequences
from evalpgm.FED.process_embeddings import process_embeddings_ESM2_35M, load_model_and_alphabet_local
from evalpgm.FED.calc_FED_during_training import calc_avg_FED
import datetime

avg_FED_list = []
stdev_FED_list = []

now = datetime.datetime.now()
date_str = now.strftime("%d_%m_%Y")
print(date_str)


class convolutional_VAE_RoPE(ModelBackbone):
    def __init__(
        self,
        vocab_size = 21, 
        hidden_sizes = [16, 32, 64, 128], 
        bottleneck_size = 128, 
        learning_rate = 0.0001, 
        blocks_per_stage = 4,
        n_transformer_layers = 5,
        n_heads = 8,
        n_warmup_steps = 1000,
        beta = 0.001,
        gamma = 1e-05,
        activation = "SwiGLU",
        use_decoder_length=True,
        eval_FED_flag = True):

        super().__init__(learning_rate = learning_rate, n_warmup_steps = n_warmup_steps)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        encoder_layers = [
            nn.Embedding(vocab_size, hidden_sizes[0]),
            Permute(0,2,1),
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)],
        ]
        for i in range(len(hidden_sizes)-1):
            encoder_layers.append(nn.Conv1d(hidden_sizes[i], hidden_sizes[i+1], kernel_size = 2, stride = 2))
            for _ in range(blocks_per_stage):
                encoder_layers.append(ResidualBlock(hidden_sizes[i+1], kernel_size= 5, dropout = 0.2))
        
        encoder_layers.append(Permute(0,2,1))
        for i in range(n_transformer_layers):
            encoder_layers.append(TransformerRoPE(hidden_sizes[-1], n_heads, dropout = 0.2, activation=activation))
        encoder_layers.append(Permute(0,2,1))

        self.encoder = nn.Sequential(*encoder_layers)
        
        self.lin_mu = nn.Conv1d(hidden_sizes[-1], bottleneck_size, kernel_size = 1)
        self.lin_var = nn.Conv1d(hidden_sizes[-1], bottleneck_size, kernel_size = 1)

        decoder_layers = []
        decoder_layers.append(nn.Conv1d(bottleneck_size, hidden_sizes[-1], kernel_size = 1))

        decoder_layers.append(Permute(0,2,1))
        for i in range(n_transformer_layers):
            decoder_layers.append(TransformerRoPE(hidden_sizes[-1], n_heads, dropout = 0.2, activation=activation))
        decoder_layers.append(Permute(0,2,1))

        for i in range(len(hidden_sizes)-1, 0, -1):
            for _ in range(blocks_per_stage):
                decoder_layers.append(ResidualBlock(hidden_sizes[i], kernel_size= 5, dropout = 0.2))
            decoder_layers.append(nn.ConvTranspose1d(hidden_sizes[i], hidden_sizes[i-1], kernel_size = 2, stride = 2))
            
        for _ in range(blocks_per_stage):
            decoder_layers.append(ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2))
        decoder_layers.append(nn.Conv1d(hidden_sizes[0], vocab_size, kernel_size = 1))

        self.decoder = nn.Sequential(*decoder_layers)

        if use_decoder_length:
            self.decoderLength = nn.Sequential(
                nn.Linear(bottleneck_size, 64),
                newGELU(),
                nn.Linear(64,32),
                newGELU(),
                nn.Linear(32,16),
                newGELU(),
                nn.Linear(16,4),
                newGELU(),
                nn.Linear(4,1),
                nn.Sigmoid()
        )
        else:
            self.decoderLength = None
        
        self.use_decoder_length = use_decoder_length


        self.eval_FED_flag = eval_FED_flag

        if self.eval_FED_flag:
            ESM2_35M, alphabet = load_model_and_alphabet_local("/home/scoutvdb/project/shared/scoutvdb/weights_ESM2/esm2_t12_35M_UR50D.pt") #not accessible idk waarom

        self.beta = beta
        self.gamma = gamma

        self.bottleneck_shape = (bottleneck_size, 1024 // 2 ** (len(hidden_sizes)-1))

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        output = self.decoder(x)
        if self.use_decoder_length:
            length_pred = self.decoderLength(x.max(-1).values)
        else:
            length_pred = None
        return output, length_pred

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
    
    def forward(self,x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x, length_pred = self.decode(z)
        return recon_x, length_pred, mean, logvar
        
    def training_step(self, batch, batch_idx):
        X, actual_lengths = batch
        if self.use_decoder_length:
            recon_x, length_pred, mean, log_var = self.forward(X) 

            counter = torch.arange(1024).expand(len(actual_lengths), -1).to(actual_lengths.device)
            mask = counter <= torch.round(actual_lengths.unsqueeze(1)*1024)

            reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[mask], X[mask])
            length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss

            self.log('train_loss', loss)
            self.log('train_KL_loss', KL_divergence)
            self.log('train_recon_loss', reconstruction_loss)
            self.log('train_KL_beta', KL_divergence * self.beta)
            self.log('train_length_loss', length_loss)
            self.log('train_length_loss_gamma', length_loss*self.gamma)

        else:
            recon_x, _, mean, log_var = self.forward(X)
            reconstruction_loss = F.cross_entropy(recon_x, X)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence

            self.log('train_loss', loss)
            self.log('train_KL_loss', KL_divergence)
            self.log('train_recon_loss', reconstruction_loss)
            self.log('train_KL_beta', KL_divergence * self.beta)

        return loss

    def test_step(self, batch, batch_idx):
        X, actual_lengths = batch
        if self.use_decoder_length:
            recon_x, length_pred, mean, log_var = self.forward(X)

            counter = torch.arange(1024).expand(len(actual_lengths), -1).to(actual_lengths.device)
            mask = counter <= torch.round(actual_lengths.unsqueeze(1)*1024)

            reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[mask], X[mask])
            length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss

            self.log('test_loss', loss)
            self.log('test_KL_loss', KL_divergence)
            self.log('test_recon_loss', reconstruction_loss)
            self.log('test_KL_beta', KL_divergence * self.beta)
            self.log('test_length_loss', length_loss)
            self.log('test_length_loss_gamma', length_loss*self.gamma)

        else:
            recon_x, _, mean, log_var = self.forward(X)
            reconstruction_loss = F.cross_entropy(recon_x, X)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence

            self.log('test_loss', loss)
            self.log('test_KL_loss', KL_divergence)
            self.log('test_recon_loss', reconstruction_loss)
            self.log('test_KL_beta', KL_divergence * self.beta)
        return loss

    def validation_step(self, batch, batch_idx): 
        X, actual_lengths = batch
        if self.use_decoder_length:
            recon_x, length_pred, mean, log_var = self.forward(X)

            counter = torch.arange(1024).expand(len(actual_lengths), -1).to(actual_lengths.device)
            mask = counter <= torch.round(actual_lengths.unsqueeze(1)*1024)

            reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[mask], X[mask])
            length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss

            self.log('validation_loss', loss)
            self.log('validation_KL_loss', KL_divergence)
            self.log('validation_recon_loss', reconstruction_loss)
            self.log('validation_KL_beta', KL_divergence * self.beta)
            self.log('validation_length_loss', length_loss)
            self.log('validation_length_loss_gamma', length_loss*self.gamma)

            if self.eval_FED_flag:
                ESM2_35M, alphabet = load_model_and_alphabet_local("/home/scoutvdb/project/shared/scoutvdb/weights_ESM2/esm2_t12_35M_UR50D.pt") 
                if batch_idx % 100000 == 0:
                    print(f"Evaluating performance of model after {batch_idx} steps. ")
                    generated_seqs, roundedLengths, avg_length = generate_sequences(model=self, amount = 2500, temperature=0.8, bottleneck_shape = self.bottleneck_shape)

                    embeddings_generated_seqs = process_embeddings_ESM2_35M(model = ESM2_35M,
                                                                                data=generated_seqs, 
                                                                                lengths=roundedLengths)
                    avg_FED, stdev_FED = calc_avg_FED(path_real_embeddings="/home/scoutvdb/project/shared/scoutvdb/data/correct_VALIDATION_esm2_t12_35M_emb.t", 
                                                    generated_embeddings=embeddings_generated_seqs, real_set_size=10000)
                    avg_FED_list.append(avg_FED)
                    stdev_FED_list.append(stdev_FED)

                    with open(f'/home/scoutvdb/project/shared/scoutvdb/{date_str}_validation_FED.txt', 'a') as f:
                        f.write(f"Gamma \t {self.gamma} \t Beta \t {self.beta} \t lr \t {self.learning_rate} \t Avg FED \t {avg_FED} \t stdev FED \t {stdev_FED} \t avg length \t {avg_length} \n")
                        f.flush()

                    self.log('avg_FED_validation', avg_FED)
                    self.log('stdev_FED_validation', stdev_FED)
                    del generated_seqs, embeddings_generated_seqs
                # Release GPU memory occupied by PyTorch tensors
                torch.cuda.empty_cache()
        else:
            recon_x, _, mean, log_var = self.forward(X)

            reconstruction_loss = F.cross_entropy(recon_x, X)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence

            self.log('validation_loss', loss)
            self.log('validation_KL_loss', KL_divergence)
            self.log('validation_recon_loss', reconstruction_loss)
            self.log('validation_KL_beta', KL_divergence * self.beta)
        return loss