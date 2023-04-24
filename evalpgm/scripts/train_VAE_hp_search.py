import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import VAE, VAE_transformer, VAE_transformer_Parallel, VAE_transformer_test_strided_conv, VAE_transformer_Parallel_test_strided_conv
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning.tuner.tuning import Tuner

#from pytorch_lightning.tuner import Tuner

#NOTE: modify training loop, see notes on desk

data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])

learning_rates = [0.00001, 0.0001, 0.001] 
betas = [0.0001, 0.001, 0.1, 1, 100]
gammas = [0.00001, 0.0001, 0.001]
#activations = ["SwiGLU", "SwiGLU_train_beta", "GeGLU", "GLU"]

models = [VAE_transformer_test_strided_conv, VAE_transformer_Parallel_test_strided_conv]
#learning_rates = [5e-4] 
#betas = [0.001]
#gammas = [0.00001]
activations = ["SwiGLU", "GeGLU"]

counter = 1

dm = Uniref50DataModule(data_path, batch_size = 1024, n_workers = 8, subsample = 0.001)
# Can also use subsample = 0.001 to use 0.1% of the data for quick testing 

for model_class in models:
    for activation in activations:
        for lr in learning_rates:
            for beta in betas:
                for gamma in gammas:
                    # VAE or use VAE_transformer or use VAE_transformer_Parallel 
                    model = model_class(
                        vocab_size = 33, 
                        hidden_sizes = [16, 32, 64, 128, 256, 256], 
                        bottleneck_size = 128, 
                        learning_rate = lr, 
                        blocks_per_stage = 2,
                        n_heads = 8,
                        n_warmup_steps = 1000,
                        beta = beta,
                        gamma=gamma,
                        activation=activation
                        )
                    
                    if model_class == VAE_transformer_test_strided_conv:
                        print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
                    elif model_class == VAE_transformer_Parallel_test_strided_conv:
                        print(" \n Using VAE_transformer_Parallel_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
                    else:
                        print("Something went wrong...")


                    callbacks = [
                        ModelCheckpoint(monitor="validation_loss", mode="min"),
                        EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
                        ]
                    if model_class == VAE_transformer_test_strided_conv:
                         name_logs = f"VAE_transformer_test_strided_conv_{activation}_lr_{lr}_beta_{beta}_gamma_{gamma}"
                    elif model_class == VAE_transformer_Parallel_test_strided_conv:
                         name_logs = f"VAE_transformer_Parallel_test_strided_conv_{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"

                    logger = TensorBoardLogger(
                        logs_path, name=name_logs
                    )

                    trainer = pl.Trainer(
                        max_steps=1500,
                        accelerator="gpu",
                        logger = logger,
                        val_check_interval=150,
                        check_val_every_n_epoch=None,
                        devices=[0], # [1] This selects the second gpu on nvidia-smi
                        callbacks=callbacks,
                        gradient_clip_val=1 #gradient clipping on norm = 1
                        )
                    
                    tuner = Tuner(trainer)
                    tuner.scale_batch_size(model, datamodule=dm)

                    #tuner = Tuner(trainer) #autoselect largest batch_size, check pytorch lightning documentation 
                    #tuner.scale_batch_size(model, datamodule=dm)

                    if model_class == VAE_transformer_test_strided_conv:
                        model_name = "VAE_transformer_test_strided_conv"
                    elif model_class == VAE_transformer_Parallel_test_strided_conv:
                         model_name = "VAE_transformer_Parallel_test_strided_conv"

                    with open('/home/scoutvdb/project/shared/scoutvdb/HP_strided_comp_180models.txt', 'a') as f:
                            f.write(f"{model_name} MODEL {counter} lr: {lr} \t beta: {beta} \t gamma: {gamma} \t activation: {activation}\n")
                            f.write("\n")
                            f.flush()

                    trainer.fit(model, dm)

                    with open('/home/scoutvdb/project/shared/scoutvdb/HP_strided_comp_180models.txt', 'a') as f:
                            f.write("__________________________________________________________________________________________________________________________________________________________________________________________________________________ \n")
                            f.flush()
                        
                    logging_file = "res_new.txt"
                    run_name = f"{model_name} {activation} lr_{lr}_beta_{beta}_gamma_{gamma}"
                    val_res = trainer.validate(
                                dataloaders=dm.val_dataloader()
                                )[0]
                        
                    with open(os.path.join(logs_path, logging_file), "a") as f:
                        f.write(
                                str(counter) + "\t" + str(model_name) + "\t" + str(activation) + "\t" +
                                "lr: \t" + str(lr) + "\t" +
                                "beta: \t" + str(beta) + "\t" +
                                "gamma: \t" + str(gamma) + "\t" +
                                "loss: \t" + format(val_res["validation_loss"], ".3f") + "\t" +        
                                "recon_loss: \t" + format(val_res["validation_reconstruction_loss"], ".3f") + "\t" +              
                                "length_loss: \t" + format(val_res["validation_length_loss"], ".3f") + "\t" +
                                "gamma*length_loss: \t" + format(val_res["validation self.gamma*length_loss"], ".3f") + "\t" +
                                "KL: \t" + format(val_res["validation_KL_divergence"], ".3f") + "\t" +
                                "beta*KL: \t" + format(val_res["validation self.beta*KL"], ".3f")  + "\n"    
                                )
                    counter += 1