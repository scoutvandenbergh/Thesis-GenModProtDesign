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
#ckpt_path = str(sys.argv[3]) 
print("Use eval_FED:", eval_FED_flag)

lr = 5e-05
model_class = convolutional_VAE_RoPE
#ckpoint_start = 100000

dm = Uniref50DataModule(data_path, batch_size = 512, n_workers = 8, subsample = 0.0005)

now = datetime.datetime.now()
date_str = now.strftime("%d_%m_%Y")
print(date_str)


betas = [0.25, 0.25, 0.5, 1, 100, 250, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.01]
gammas = [1, 500, 100, 1, 25, 500, 25, 50, 100, 250, 500, 100, 250, 50, 100, 250, 500, 500]

for beta in betas:
    for gamma in gammas:
        model = model_class(
            vocab_size = 33, 
            hidden_sizes = [16, 32, 64, 128, 256, 256], 
            bottleneck_size = 128, 
            learning_rate = 5e-05,
            blocks_per_stage = 2,
            n_heads = 8,
            n_warmup_steps = 1000,
            beta = beta,
            gamma=gamma,
            activation="SwiGLU",
            use_decoder_length=True,
            eval_FED_flag = eval_FED_flag
        )                

        callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint"),  # Save the best checkpoint based on validation loss
        ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=2500, filename="{step}"),
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=100, filename="checkpoint_{step}"), # Save a checkpoint every 100 steps, overwriting the previous one
        EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
        ]


        logger = TensorBoardLogger(
            logs_path, name=f"{date_str}_convolutional_VAE_RoPE_lr_{lr}_beta_{beta}_gamma_{gamma}"
        )

        trainer = pl.Trainer(
            max_steps=250,
            accelerator="gpu",
            logger = logger,
            val_check_interval=200,
            check_val_every_n_epoch=None,
            devices=[0], # [1] This selects the second gpu on nvidia-smi, not required for HPC? --> [0] for HPC
            callbacks=callbacks,
            gradient_clip_val=1 #gradient clipping on norm = 1
        )

        trainer.fit(model, dm)

print("Trainer done")