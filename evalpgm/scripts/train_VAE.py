import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import VAE, VAE_transformer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.tuner import Tuner

data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])

learning_rates = [0.00001, 0.0001, 0.001] 
betas = [0.0001, 0.001, 0.1, 1, 100]
gammas = [0.00001, 0.0001, 0.001]
activations = ["SwiGLU", "GeGLU", "GLU"]
beta_swiglu = [None, 1]

# learning_rates = [5e-4] 
# betas = [0.001]
# gammas = [0.00001]
activations = ["SwiGLU"]

counter = 1

dm = Uniref50DataModule(data_path, batch_size = 256, n_workers = 8, subsample = 0.001)
# Can also use subsample = 0.001 to use 0.1% of the data for quick testing

for beta_swi in beta_swiglu:
    for activation in activations:
        for lr in learning_rates:
            for beta in betas:
                for gamma in gammas:
                    # model or use VAE_transformer
                    model = VAE_transformer(
                        vocab_size = 33, 
                        hidden_sizes = [16, 32, 64, 128, 256], 
                        bottleneck_size = 128, 
                        learning_rate = lr, 
                        blocks_per_stage = 2,
                        n_heads = 8,
                        n_warmup_steps = 1000,
                        beta = beta,
                        gamma=gamma,
                        beta_swi=beta_swi,
                        activation=activation
                        )
                    print(" \n Using ", activation, " in feedforward layer of Transformer with beta_swiglu ", beta_swi, ". \n")

                    callbacks = [
                        ModelCheckpoint(monitor="validation_loss", mode="min"),
                        EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
                        ]
                    logger = TensorBoardLogger(
                        logs_path, name=f"{activation} beta_swi {beta_swi} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
                    )

                    trainer = pl.Trainer(
                        max_steps=5000,
                        accelerator="gpu",
                        logger = logger,
                        val_check_interval=25,
                        check_val_every_n_epoch=None,
                        devices=[1], # [1] This selects the second gpu on nvidia-smi
                        callbacks=callbacks,
                        gradient_clip_val=1 #gradient clipping on norm = 1
                        )

                    #tuner = Tuner(trainer) #autoselect largest batch_size, check pytorch lightning documentation
                    #tuner.scale_batch_size(model, datamodule=dm)

                    with open('testing SwiGLU.txt', 'a') as f:
                            f.write(f"num_blocks_2 0_001 subsample MODEL {counter} lr: {lr} \t beta: {beta} \t gamma: {gamma} \t activation: {activation} \t beta_swi: {beta_swi}\n")
                            f.write("\n")
                            f.flush()

                    trainer.fit(model, dm)
                    counter += 1

                    with open('testing SwiGLU.txt', 'a') as f:
                            f.write("__________________________________________________________________________________________________________________________________________________________________________________________________________________ \n")
                            f.flush()