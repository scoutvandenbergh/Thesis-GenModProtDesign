import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from evalpgm.blocks import Permute, View, ResidualBlock, ModelBackbone, TransformerRoPE, TransformerRoPE_Parallel, newGELU
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
        hidden_sizes = [16, 32, 64, 128, 256, 512], 
        bottleneck_size = 128, 
        learning_rate = 0.0001, 
        blocks_per_stage = 4,
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
        

        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_sizes[0]), # gives B x 1024 x 16 BxLxC 
            Permute(0,2,1), # B x 16 x 1024 BxCxL
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], #B x 16 x 1024
            nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size = 4, stride = 4), # B x 32 x 256
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 32 x 256
            nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size = 2, stride = 2), # B x 64 x 128
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 64 x 128
            nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size = 2, stride = 2), # B x 128 x 64
            Permute(0,2,1), # B x L x C
            *[TransformerRoPE(hidden_sizes[3], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], # B x 64 x 128 BxLxC
            Permute(0,2,1), # BxCxL
            nn.Conv1d(hidden_sizes[3], hidden_sizes[4], kernel_size = 2, stride = 2), # B x 256 x 32
            Permute(0,2,1), # B x 32 x 256
            *[TransformerRoPE(hidden_sizes[4], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], # B x 32 x 256
            Permute(0,2,1), # B x 256 x 32 BxCxL
            
            nn.Conv1d(hidden_sizes[4], hidden_sizes[5], kernel_size = 2, stride = 2), # B x 512 x 16
            Permute(0,2,1), # B x 16 x 512
            *[TransformerRoPE(hidden_sizes[5], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], # B x 16 x 512
            Permute(0,2,1), # B x 512 x 16 BxCxL

            View(-1, hidden_sizes[5]*16),
            nn.Linear(hidden_sizes[5]*16, bottleneck_size),
            newGELU()
        )
        
        self.lin_mu = nn.Linear(bottleneck_size, bottleneck_size)
        self.lin_var = nn.Linear(bottleneck_size, bottleneck_size)

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_sizes[5]*16), #Bx8192
            newGELU(),
            View(-1, hidden_sizes[5], 16), # put them into B x 512 x 16, just as it was in last conv of encoder
            Permute(0,2,1), # BxLxC
            *[TransformerRoPE(hidden_sizes[5], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], #Bx16x512
            Permute(0,2,1), #Bx512x16
            nn.ConvTranspose1d(hidden_sizes[5], hidden_sizes[4], kernel_size = 2, stride = 2), # Bx256x32
            Permute(0,2,1), #Bx32x256

            *[TransformerRoPE(hidden_sizes[4], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], #Bx32x256
            Permute(0,2,1), #Bx256x32
            nn.ConvTranspose1d(hidden_sizes[4], hidden_sizes[3], kernel_size = 2, stride = 2), # Bx128x64
            Permute(0,2,1), #BxLxC Bx64x128

            *[TransformerRoPE(hidden_sizes[3], n_heads, dropout = 0.2, activation=activation) for _ in range(blocks_per_stage)], #Bx64x128
            Permute(0,2,1), # BxCxL Bx128x64
            nn.ConvTranspose1d(hidden_sizes[3], hidden_sizes[2], kernel_size = 2, stride = 2), # B x 64 x 256
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], #Bx64x256
            nn.ConvTranspose1d(hidden_sizes[2], hidden_sizes[1], kernel_size = 2, stride = 2), # Bx32x512
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], #Bx32x512
            nn.ConvTranspose1d(hidden_sizes[1], hidden_sizes[0], kernel_size = 4, stride = 4), # Bx16x1024
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], #Bx16x1024
            nn.Conv1d(hidden_sizes[0], vocab_size, kernel_size = 1), # to output B x 21 x 1024
        )

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

    def encode(self, x):
        x = self.encoder(x)
        return self.lin_mu(x), self.lin_var(x)
    
    def decode(self, x):
        output = self.decoder(x)
        if self.use_decoder_length:
            length_pred = self.decoderLength(x)
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
            mask = counter <= torch.round(actual_lengths.unsqueeze(1)*1024) #moet dit * 1024???

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
                #niet ideaal hier maar anders not accessed, en kan anders ook niet gecontroleerd aan/uitzetten via sys.argv 
                if batch_idx % 100000 == 0:
                    print(f"Evaluating performance of model after {batch_idx} steps. ")
                    generated_seqs, roundedLengths, avg_length = generate_sequences(model=self, amount = 2500, temperature=0.8)

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
            #lengthmask = self.get_mask_full_length(actual_lengths)

            #reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[lengthmask], X[lengthmask])
            print("X", X.shape)
            print("recon_x", recon_x.shape)
            print("recon_x permute(0,2,1)", recon_x.permute(0,2,1).shape)
            reconstruction_loss = F.cross_entropy(recon_x, X)
            KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

            loss = reconstruction_loss + self.beta * KL_divergence

            self.log('validation_loss', loss)
            self.log('validation_KL_loss', KL_divergence)
            self.log('validation_recon_loss', reconstruction_loss)
            self.log('validation_KL_beta', KL_divergence * self.beta)

            if batch_idx % 1000 == 0:
                with open('/home/scoutvdb/project/shared/scoutvdb/26_april_NO_decoderLength_hp_decoderLength_stridedConv_model.txt', 'a') as f:
                    indices = random.sample(range(X.shape[0]), min(X.shape[0], 1))
                    for i in indices:
                        f.write(f'[VALIDATION] Epoch {self.current_epoch}, Step {batch_idx}: Predicted: {round(torch.tensor(length_pred[i]).item()*1024, 2)}, Target: {round(torch.tensor(actual_lengths[i]).item()*1024, 2)}, (MSE, self.gamma*MSE): {F.mse_loss(length_pred[i], actual_lengths[i]).item()}, Recon: {round(torch.tensor(reconstruction_loss).item(), 2)}, KL: {round(self.beta * torch.tensor(KL_divergence).item(), 2)}, loss: {round(loss.item(), 2)}\n')
                        f.flush()

        return loss

    def get_mask(self, lengths):
        lengths_clamped = torch.clamp(lengths.detach().squeeze(), min=64, max=1024) # B
        counter = torch.arange(1024).expand(len(lengths_clamped), -1).to(lengths_clamped.device)
        mask = counter <= lengths_clamped.unsqueeze(1) # B x 1024
        return mask

    def get_mask_full_length(self, lengths):
        counter = torch.arange(1024).expand(len(lengths), -1).to(lengths.device)
        mask = counter < lengths.unsqueeze(1) # B x 1024
        return mask
    

class VAE(ModelBackbone):
    def __init__(
        self,
        vocab_size = 21, 
        hidden_sizes = [64, 128, 256, 512, 1024], 
        bottleneck_size = 128, 
        learning_rate = 0.0001, 
        blocks_per_stage = 4,
        #n_heads = 8,
        n_warmup_steps = 1000,
        beta = 0.001,
        gamma = 1e-05):

        super().__init__(learning_rate = learning_rate, n_warmup_steps = n_warmup_steps)
        self.save_hyperparameters()
        

        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_sizes[0]), # gives B x 1024 x 64
            Permute(0,2,1), # B x 64 x 1024
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 64 x 1024
            nn.Conv1d(hidden_sizes[0], hidden_sizes[1], kernel_size = 4, stride = 4), # B x 128 x 256
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 128 x 256
            nn.Conv1d(hidden_sizes[1], hidden_sizes[2], kernel_size = 4, stride = 4), # B x 256 x 64
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 256 x 64
            nn.Conv1d(hidden_sizes[2], hidden_sizes[3], kernel_size = 4, stride = 4), # B x 512 x 16
            *[ResidualBlock(hidden_sizes[3], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 512 x 16
            nn.Conv1d(hidden_sizes[3], hidden_sizes[4], kernel_size = 4, stride = 4), # B x 1024 x 4
            *[ResidualBlock(hidden_sizes[4], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 1024 x 4
            View(-1, hidden_sizes[4]*4), # B x 1024*4
            nn.Linear(hidden_sizes[4]*4, bottleneck_size), # B x 128 
            newGELU()
        )
        
        self.lin_mu = nn.Linear(bottleneck_size, bottleneck_size) 
        self.lin_var = nn.Linear(bottleneck_size, bottleneck_size)

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_sizes[4]*4), # B x 1024*4 
            newGELU(),
            View(-1, hidden_sizes[4], 4), # B x 1024 x 4
            *[ResidualBlock(hidden_sizes[4], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 1024 x 4
            nn.ConvTranspose1d(hidden_sizes[4], hidden_sizes[3], kernel_size = 4, stride = 4), # B x 512 x 16
            *[ResidualBlock(hidden_sizes[3], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 512 x 16
            nn.ConvTranspose1d(hidden_sizes[3], hidden_sizes[2], kernel_size = 4, stride = 4), # B x 256 x 64
            *[ResidualBlock(hidden_sizes[2], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 256 x 64
            nn.ConvTranspose1d(hidden_sizes[2], hidden_sizes[1], kernel_size = 4, stride = 4), # B x 128 x 256
            *[ResidualBlock(hidden_sizes[1], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 128 x 256
            nn.ConvTranspose1d(hidden_sizes[1], hidden_sizes[0], kernel_size = 4, stride = 4), # B x 64 x 1024
            *[ResidualBlock(hidden_sizes[0], kernel_size= 5, dropout = 0.2) for _ in range(blocks_per_stage)], # B x 64 x 1024
            nn.Conv1d(hidden_sizes[0], vocab_size, kernel_size = 1), # B x 21 x 1024
        )

        self.decoderLength = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            newGELU(),
            nn.Linear(64,32),
            newGELU(),
            nn.Linear(32,16),
            newGELU(),
            nn.Linear(16,4),
            newGELU(),
            nn.Linear(4,1)
        )

        self.beta = beta
        self.gamma = gamma

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
        return self.decode(z), mean, logvar , self.decoderLength(z) 
        
    def training_step(self, batch, batch_idx):
        X, actual_lengths = batch
        recon_x, mean, log_var, length_pred = self.forward(X)

        lengthmask = self.get_mask(length_pred)

        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[lengthmask], X[lengthmask])
        length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        X, actual_lengths = batch
        recon_x, mean, log_var, length_pred = self.forward(X)

        lengthmask = self.get_mask(length_pred)

        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[lengthmask], X[lengthmask])
        length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss
        self.log('test_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        X, actual_lengths = batch
        recon_x, mean, log_var, length_pred = self.forward(X)
        
        lengthmask = self.get_mask(length_pred)

        reconstruction_loss = F.cross_entropy(recon_x.permute(0,2,1)[lengthmask], X[lengthmask])
        length_loss = F.mse_loss(length_pred.squeeze(), actual_lengths)
        KL_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

        loss = reconstruction_loss + self.beta * KL_divergence + self.gamma * length_loss
        self.log('validation_reconstruction_loss', reconstruction_loss)
        self.log('validation_KL_divergence', KL_divergence)
        self.log('validation_length_loss', length_loss)
        self.log('validation_loss', loss)
        return loss

    def get_mask(self, lengths):
        lengths_clamped = torch.clamp(lengths.detach().squeeze(), min=64, max=1024) # B
        counter = torch.arange(1024).expand(len(lengths_clamped), -1).to(lengths_clamped.device)
        mask = counter < lengths_clamped.unsqueeze(1) # B x 1024
        return mask
