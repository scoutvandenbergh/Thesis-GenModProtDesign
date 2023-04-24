import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import VAE, VAE_transformer, VAE_transformer_Parallel, VAE_transformer_test_strided_conv, avg_FED_list, stdev_FED_list
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os

data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])

lr = 0.001
beta = 0.1
gamma = 1e-05
activation = "SwiGLU"
model_class = VAE_transformer_test_strided_conv

counter = 0

dm = Uniref50DataModule(data_path, batch_size = 256, n_workers = 8, subsample = 0.001)

model = model_class(
    vocab_size = 33, 
    hidden_sizes = [16, 32, 64, 128, 256, 256], #[16, 32, 64, 128, 256, 256] gives 11.1M params; [16, 32, 64, 128, 256, 512] gives 25M params
    bottleneck_size = 128,
    learning_rate = lr, 
    blocks_per_stage = 2,
    n_heads = 8,
    n_warmup_steps = 1000,
    beta = beta,
    gamma=gamma,
    activation=activation
)
                    
if model_class == VAE_transformer:
    print(" \n Using VAE_transformer with " , activation, " in feedforward layer of Transformer \n")
elif model_class == VAE_transformer_Parallel:
    print(" \n Using VAE_transformer_Parallel with " , activation, " in feedforward layer of Transformer \n")
elif model_class == VAE_transformer_test_strided_conv:
    print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
else:
    print("Something went wrong...")


callbacks = [
    ModelCheckpoint(monitor="validation_loss", mode="min"),
    EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
]
if model_class == VAE_transformer:
    name_logs = f"testFED_VAE_transformer {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
elif model_class == VAE_transformer_Parallel:
    name_logs = f"test_FEDVAE_transformer_Parallel {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
elif model_class == VAE_transformer_test_strided_conv:
    name_logs = f"testFED_checkpoint_VAE_transformer_strided_conv"

logger = TensorBoardLogger(
    logs_path, name=name_logs
)

trainer = pl.Trainer(
    max_steps=500,
    accelerator="gpu",
    logger = logger,
    val_check_interval=50,
    check_val_every_n_epoch=None,
    devices=[1], # [1] This selects the second gpu on nvidia-smi, not required for HPC?
    callbacks=callbacks,
    gradient_clip_val=1 #gradient clipping on norm = 1
)


if model_class == VAE_transformer:
    model_name = "VAE_transformer"
elif model_class == VAE_transformer_Parallel:
    model_name = "VAE_transformer_Parallel"
elif model_class == VAE_transformer_test_strided_conv:
    model_name = "VAE_transformer_test_strided_conv"

with open('/home/scoutvdb/project/shared/scoutvdb/TEST_FED_strided_conv_optimized_transformer vs transformer_parallel.txt', 'a') as f:
    f.write(f"TEST_FED_optimized_{model_name} num_blocks_2 0_001 subsample MODEL {counter} lr: {lr} \t beta: {beta} \t gamma: {gamma} \t activation: {activation}\n")
    f.write("\n")
    f.flush()

trainer.fit(model, dm)

with open('/home/scoutvdb/project/shared/scoutvdb/TEST_FED_strided_conv_optimized_transformer vs transformer_parallel.txt', 'a') as f:
    f.write("__________________________________________________________________________________________________________________________________________________________________________________________________________________ \n")
    f.flush()
                        
logging_file = "resTEST_FED.txt"
run_name = f"TEST_FED_optimized_{model_name} {activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
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

print(avg_FED_list)
print(stdev_FED_list)