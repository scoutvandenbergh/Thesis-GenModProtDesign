
import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import convolutional_VAE_RoPE
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import datetime


data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])
eval_FED_flag = str(sys.argv[3]).lower() == 'true'
print("Use eval_FED:", eval_FED_flag)

model_class = convolutional_VAE_RoPE

dm = Uniref50DataModule(data_path, batch_size = 32, n_workers = 8, subsample = 0.001)

now = datetime.datetime.now()
date_str = now.strftime("%d_%m_%Y")
print(date_str)

lr = 1e-04
betas = [0.01]
gammas = [100]

for beta in betas:
    for gamma in gammas:
        print(f"Using beta {beta}, gamma {gamma}")
        model = model_class(
            vocab_size = 33, 
            hidden_sizes = [256, 256, 256, 256, 256, 256, 256], 
            bottleneck_size = 8, 
            learning_rate = lr,
            blocks_per_stage = 6,
            n_transformer_layers = 6,
            n_heads = 8,
            n_warmup_steps = 100,
            beta = beta,
            gamma=gamma,
            activation="SwiGLU",
            use_decoder_length=True, #put on True if using gamma
            eval_FED_flag = eval_FED_flag
        )

        callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint"),
        ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=5000, filename="{step}"),
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=500, filename="checkpoint_{step}"),
        EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
        ]


        logger = TensorBoardLogger(
            logs_path, name=f"{date_str}_convolutional_VAE_RoPE_lr_{lr}_beta_{beta}_gamma_{gamma}_batch_size_32_L_16_layers_6"
        )

        trainer = pl.Trainer(
            max_steps=50000,
            accelerator="gpu",
            logger = logger,
            val_check_interval=10000,
            check_val_every_n_epoch=None,
            devices=[0],
            callbacks=callbacks,
            gradient_clip_val=1
        )

        trainer.fit(model, dm)

print("Trainer done")