import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models_train import VAE, VAE_transformer, VAE_transformer_Parallel, VAE_transformer_test_strided_conv
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os

data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])
#ckpt_path = str(sys.argv[3])

lr = 0.0001
activation = "SwiGLU"
model_class = VAE_transformer_test_strided_conv
ckpoint_start = 100000

counter = 0

dm = Uniref50DataModule(data_path, batch_size = 512, n_workers = 8, subsample = None)

model = model_class(
    vocab_size = 33, 
    hidden_sizes = [16, 32, 64, 128, 256, 256], 
    bottleneck_size = 128, 
    learning_rate = lr,
    blocks_per_stage = 2,
    n_heads = 8,
    n_warmup_steps = 1000,
    beta = 0.001,
    gamma=10,
    activation="SwiGLU",
    use_decoder_length=True
)
                    
# model.load_from_checkpoint(ckpt_path)

if model_class == VAE_transformer:
    print(" \n Using VAE_transformer with " , activation, " in feedforward layer of Transformer \n")
elif model_class == VAE_transformer_Parallel:
    print(" \n Using VAE_transformer_Parallel with " , activation, " in feedforward layer of Transformer \n")
elif model_class == VAE_transformer_test_strided_conv:
    print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
else:
    print("Something went wrong...")


callbacks = [
ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint"),  # Save the best checkpoint based on validation loss
ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=10000, filename="{step}"),
ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=100, filename="checkpoint_{step}"), # Save a checkpoint every 100 steps, overwriting the previous one
EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
]

# if model_class == VAE_transformer:
#     name_logs = f"testFED_VAE_transformer {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
# elif model_class == VAE_transformer_Parallel:
#     name_logs = f"test_FEDVAE_transformer_Parallel {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
if model_class == VAE_transformer_test_strided_conv:
    name_logs = f"TESTTTTT_wrong_recon_loss_VAE_from_checkpoint_start_{ckpoint_start}"

logger = TensorBoardLogger(
    logs_path, name=name_logs
)

trainer = pl.Trainer(
    max_steps=1000,
    accelerator="gpu",
    logger = logger,
    val_check_interval=1000,
    check_val_every_n_epoch=None,
    devices=[0], # [1] This selects the second gpu on nvidia-smi, not required for HPC? --> [0] for HPC
    callbacks=callbacks,
    gradient_clip_val=1 #gradient clipping on norm = 1
)


if model_class == VAE_transformer:
    model_name = "VAE_transformer"
elif model_class == VAE_transformer_Parallel:
    model_name = "VAE_transformer_Parallel"
elif model_class == VAE_transformer_test_strided_conv:
    model_name = "VAE_transformer_test_strided_conv"

# with open('/home/scoutvdb/project/shared/scoutvdb/TEST_FED_strided_conv_optimized_transformer vs transformer_parallel.txt', 'a') as f:
#     f.write(f"TEST_FED_optimized_{model_name} num_blocks_2 0_001 subsample MODEL {counter} lr: {lr} \t beta: {beta} \t gamma: {gamma} \t activation: {activation}\n")
#     f.write("\n")
#     f.flush()

trainer.fit(model, dm)

# with open('/home/scoutvdb/project/shared/scoutvdb/TEST_FED_strided_conv_optimized_transformer vs transformer_parallel.txt', 'a') as f:
#     f.write("__________________________________________________________________________________________________________________________________________________________________________________________________________________ \n")
#     f.flush()
                        
# logging_file = "resTEST_FED.txt"
# run_name = f"TEST_FED_optimized_{model_name} {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
# val_res = trainer.validate(
#     dataloaders=dm.val_dataloader()
#     )[0]
                        
# with open(os.path.join(logs_path, logging_file), "a") as f:
#     f.write(
#     str(counter) + "\t" + str(model_name) + "\t" + str(activation) + "\t" +
#     "lr: \t" + str(lr) + "\t" +
#     "beta: \t" + str(beta) + "\t" +
#     "gamma: \t" + str(gamma) + "\t" +
#     "loss: \t" + format(val_res["validation_loss"], ".3f") + "\t" +        
#     "recon_loss: \t" + format(val_res["validation_reconstruction_loss"], ".3f") + "\t" +              
#     "length_loss: \t" + format(val_res["validation_length_loss"], ".3f") + "\t" +
#     "gamma*length_loss: \t" + format(val_res["validation self.gamma*length_loss"], ".3f") + "\t" +
#     "KL: \t" + format(val_res["validation_KL_divergence"], ".3f") + "\t" +
#     "beta*KL: \t" + format(val_res["validation self.beta*KL"], ".3f")  + "\n"    
# )

counter += 1
