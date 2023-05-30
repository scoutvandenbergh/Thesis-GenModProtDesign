import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import VAE, VAE_transformer, VAE_transformer_Parallel, VAE_transformer_test_strided_conv, VAE_transformer_Parallel_test_strided_conv
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning.tuner.tuning import Tuner
import datetime
import time
import torch

start_time = time.time()


data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])

#learning_rates = [0.00001, 0.0001, 0.001] 
#betas = [0.0001, 0.001, 0.1, 1, 100]
#gammas = [1]
#gammas = [0.00001, 0.0001, 0.001]
#activations = ["SwiGLU", "SwiGLU_train_beta", "GeGLU", "GLU"]

#models = [VAE_transformer_test_strided_conv, VAE_transformer_Parallel_test_strided_conv]

#learning_rates = [5e-4] 
#betas = [0.001]
#gammas = [0.00001]

#learning_rates = [0.001, 0.0001]
#betas = [0.001]
#gammas = [0.1, 1, 10, 100] in 26_april_new_find_gamma
#gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# learning_rates = [0.00005]
#betas = [0.001]
#gammas = [10]
# betas = [0.0001, 0.005, 0.001, 0.05]
# gammas = [10, 50, 100, 500]
# betas = [0.1, 0.5, 1, 10, 50, 250]
# gammas = [10, 50, 100, 250]
# gammas = [10, 25, 50, 100, 250]
# betas = [0.2, 0.3, 0.4]


# hyperparameters = {
#     "model_1_12_may": {"learning_rate": 0.0001, "beta": 0.0001, "gamma": 500, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_0.0001_beta_0.0001_gamma_500/version_0/checkpoints/step_10000.ckpt"},
#     "model_2_12_may": {"learning_rate": 0.00005, "beta": 0.0001, "gamma": 50, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_50/version_0/checkpoints/step_10000.ckpt"},
#     "model_3_12_may": {"learning_rate": 0.00005, "beta": 0.005, "gamma": 100, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.005_gamma_100/version_0/checkpoints/step_10000.ckpt"},
#     "model_4_12_may_lowest_FED_first_10k_steps": {"learning_rate": 0.00005, "beta": 0.001, "gamma": 10, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_10/version_0/checkpoints/step_10000.ckpt"},
#     "model_5_12_may": {"learning_rate": 0.00005, "beta": 0.001, "gamma": 50, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_50/version_0/checkpoints/step_10000.ckpt"},
#     "model_6_12_may": {"learning_rate": 0.00005, "beta": 0.001, "gamma": 500, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_500/version_0/checkpoints/step_10000.ckpt"},
#     "model_7_12_may": {"learning_rate": 0.00005, "beta": 0.3, "gamma": 25, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/16_may_hp_search_FED_val_set_logs/new_16_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.3_gamma_25/version_0/checkpoints/step_10000.ckpt"},

#     # and so on for all models
# }

# hyperparameters = {
#     "model_1_17_may": {"learning_rate": 0.00005, "beta": 0.0001, "gamma": 50, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_50/version_0/checkpoints/step_10000.ckpt"},
#     "model_2_17_may": {"learning_rate": 0.00005, "beta": 0.005, "gamma": 100, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.005_gamma_100/version_0/checkpoints/step_10000.ckpt"},
#     "model_3_17_may": {"learning_rate": 0.00005, "beta": 0.001, "gamma": 10, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_10/version_0/checkpoints/step_10000.ckpt"},
#     "model_4_17_may": {"learning_rate": 0.00005, "beta": 0.001, "gamma": 500, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_500/version_0/checkpoints/step_10000.ckpt"},
#     "model_5_17_may": {"learning_rate": 0.00005, "beta": 0.3, "gamma": 25, "pretrained_model_path": "/home/scoutvdb/project/shared/scoutvdb/logs/16_may_hp_search_FED_val_set_logs/new_16_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.3_gamma_25/version_0/checkpoints/step_10000.ckpt"},

#     # and so on for all models
# }

betas = [0.25, 0.25, 0.5, 1, 100, 250, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.01]
gammas = [1, 500, 100, 1, 25, 500, 25, 50, 100, 250, 500, 100, 250, 50, 100, 250, 500, 500]

pretrained_model_paths = ["/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.25_gamma_1/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.25_gamma_500/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.5_gamma_100/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_1_gamma_1/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_100_gamma_25/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_250_gamma_500/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_25/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_50/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_100/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_250/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.0001_gamma_500/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.005_gamma_100/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.005_gamma_250/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_50/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_100/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_250/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_500/version_0/checkpoints/checkpoint_step=10000.ckpt",
                          "/home/scoutvdb/project/shared/scoutvdb/logs/24_may_hp_search_FED_reconstruction_fixed/24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.01_gamma_500/version_0/checkpoints/checkpoint_step=10000.ckpt"
]

models_dict = {}


for i in range(18):
    model_name = f"model_{i+1}"
    models_dict[model_name] = {
        "beta": betas[i],
        "gamma": gammas[i],
        "pretrained_model_path": pretrained_model_paths[i]
    }

print(models_dict)

batch_size = 256
dm = Uniref50DataModule(data_path, batch_size = batch_size, n_workers = 8, subsample = None)

counter = 1

# Can also use subsample = 0.001 to use 0.1% of the data for quick testing 

# # Iterate over the models and their hyperparameters
for model_name, params in models_dict.items():
    beta = params["beta"]
    gamma = params["gamma"]
    pretrained_model_path = params["pretrained_model_path"]

    model = VAE_transformer_test_strided_conv.load_from_checkpoint(
        checkpoint_path=pretrained_model_path, 
        vocab_size=33,
        hidden_sizes=[16, 32, 64, 128, 256, 256],
        bottleneck_size=128,
        learning_rate=5e-05,
        blocks_per_stage=2,
        n_heads=8,
        n_warmup_steps=1000,
        beta=beta,
        gamma=gamma,
        activation="SwiGLU",
        use_decoder_length=True
    )

    # if model_class == VAE_transformer_test_strided_conv:
    #     print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
    # elif model_class == VAE_transformer_Parallel_test_strided_conv:
    #     print(" \n Using VAE_transformer_Parallel_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
    # elif model_class == VAE_transformer:
    #     print(" \n Using VAE_transformer with " , activation, " in feedforward layer of Transformer \n"
    # else:
    #     print("Something went wrong...")

    print(f"Using gamma as {gamma} and a batch size of {batch_size} beta {beta} lr 5e-05\n")

    callbacks = [
        ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint_{step}"),  # Save the best checkpoint based on validation loss
        ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=5000, filename="{step}"),
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=100, filename="checkpoint_{step}"), # Save a checkpoint every 100 steps, overwriting the previous one
        EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
    ]
    # if model_class == VAE_transformer_test_strided_conv:
    #         name_logs = f"new_16_may_VAE_transformer_test_strided_conv_{activation}_lr_{lr}_beta_{beta}_gamma_{gamma}"
    # elif model_class == VAE_transformer_Parallel_test_strided_conv:
    #         name_logs = f"TEST_new_VAE_transformer_Parallel_test_strided_conv_{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
    # elif model_class == VAE_transformer:
    #         name_logs = f"TEST_new_VAE_transformer{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
                                    

    logger = TensorBoardLogger(
        logs_path, name=f"29_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_{beta}_gamma_{gamma}"
    )

    trainer = pl.Trainer(
        max_steps=40000,
        accelerator="gpu",
        logger = logger,
        val_check_interval=5000,
        check_val_every_n_epoch=None,
        devices=[0], # [1] This selects the second gpu on nvidia-smi
        callbacks=callbacks,
        gradient_clip_val=1 #gradient clipping on norm = 1
        )
                                
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, datamodule=dm)

    # if model_class == VAE_transformer_test_strided_conv:
    #     model_name = "VAE_transformer_test_strided_conv"
    # elif model_class == VAE_transformer_Parallel_test_strided_conv:
    #     model_name = "VAE_transformer_Parallel_test_strided_conv"
    # elif model_class == VAE_transformer:
    #     model_name = "VAE_transformer_test_DecoderLength_hp"

    trainer.fit(model, dm)
                                    
    logging_file = "29_may_working_recon_error_hopefully_testFED_res.txt"
    run_name = f"VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_{beta}_gamma_{gamma}"
    val_res = trainer.validate(
                dataloaders=dm.val_dataloader()
                )[0]
                        
with open(os.path.join(logs_path, logging_file), "a") as f:
        f.write(
                str(counter) + "\t" + "VAE_transformer_test_strided_conv" + "\t" + "SwiGLU" + "\t" +
                "lr: \t" + str("5e-05") + "\t" +
                "beta: \t" + str(beta) + "\t" +
                "gamma: \t" + str(gamma) + "\t" +
                "loss: \t" + format(val_res["validation_loss"], ".3f") + "\t" +        
                "recon_loss: \t" + format(val_res["validation_recon_loss"], ".3f") + "\t" +              
                "length_loss: \t" + format(val_res["validation_length_loss"], ".3f") + "\t" +
                "gamma*length_loss: \t" + format(val_res["validation_length_loss_gamma"], ".3f") + "\t" +
                "KL: \t" + format(val_res["validation_KL_loss"], ".3f") + "\t" +
                "beta*KL: \t" + format(val_res["validation_KL_beta"], ".3f")  + "\n"    
                )
counter += 1
stop_time = time.time()
print("Time elapsed ", datetime.timedelta(seconds=stop_time-start_time))

# learning_rates = [5e-05] 
# betas = [0.1, 0.25, 0.5, 1, 50, 100, 250]
# # gammas = [1, 10, 25, 50, 100, 250, 500]

# # betas = [0.0001, 0.005, 0.001, 0.01]
# gammas = [1, 10, 25, 50, 100, 250, 500]

# activations = ["SwiGLU"]
# batch_size = 256
# dm = Uniref50DataModule(data_path, batch_size = batch_size, n_workers = 8, subsample = 0.01)

# for lr in learning_rates:
#     for beta in betas:
#         for gamma in gammas:
#             model = VAE_transformer_test_strided_conv(
#                 vocab_size=33,
#                 hidden_sizes=[16, 32, 64, 128, 256, 256],
#                 bottleneck_size=128,
#                 learning_rate=lr,
#                 blocks_per_stage=2,
#                 n_heads=8,
#                 n_warmup_steps=1000,
#                 beta=beta,
#                 gamma=gamma,
#                 activation="SwiGLU",
#                 use_decoder_length=True
#             )

#             # VAE or use VAE_transformer or use VAE_transformer_Parallel 
#             # model = model_class(
#             #     vocab_size = 33, 
#             #     hidden_sizes = [16, 32, 64, 128, 256, 256], 
#             #     bottleneck_size = 128, 
#             #     learning_rate = lr, 
#             #     blocks_per_stage = 2,
#             #     n_heads = 8,
#             #     n_warmup_steps = 1000,
#             #     beta = beta,
#             #     gamma=gamma,
#             #     activation=activation,
#             #     use_decoder_length=True
#             #     )
#             # model.load_state_dict(torch.load(pretrained_model_path))

                                
#             # if model_class == VAE_transformer_test_strided_conv:
#             #     print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
#             # elif model_class == VAE_transformer_Parallel_test_strided_conv:
#             #     print(" \n Using VAE_transformer_Parallel_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
#             # elif model_class == VAE_transformer:
#             #     print(" \n Using VAE_transformer with " , activation, " in feedforward layer of Transformer \n")

#             # else:
#             #     print("Something went wrong...")

#             print(f"Using gamma as {gamma} and a batch size of {batch_size} beta {beta} lr {lr}\n")

#             callbacks = [
#                 ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint_{step}"),  # Save the best checkpoint based on validation loss
#                 ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=1000, filename="{step}"),
#                 ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=100, filename="checkpoint_{step}"), # Save a checkpoint every 100 steps, overwriting the previous one
#                 EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
#             ]
#             # if model_class == VAE_transformer_test_strided_conv:
#             #         name_logs = f"new_16_may_VAE_transformer_test_strided_conv_{activation}_lr_{lr}_beta_{beta}_gamma_{gamma}"
#             # elif model_class == VAE_transformer_Parallel_test_strided_conv:
#             #         name_logs = f"TEST_new_VAE_transformer_Parallel_test_strided_conv_{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
#             # elif model_class == VAE_transformer:
#             #         name_logs = f"TEST_new_VAE_transformer{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
                                    

#             logger = TensorBoardLogger(
#                 logs_path, name=f"24_may_VAE_transformer_test_strided_conv_SwiGLU_lr_{lr}_beta_{beta}_gamma_{gamma}"
#             )

#             trainer = pl.Trainer(
#                 max_steps=10000,
#                 accelerator="gpu",
#                 logger = logger,
#                 val_check_interval=1000,
#                 check_val_every_n_epoch=None,
#                 devices=[1], # [1] This selects the second gpu on nvidia-smi
#                 callbacks=callbacks,
#                 gradient_clip_val=1 #gradient clipping on norm = 1
#                 )
                                
#             #tuner = Tuner(trainer)
#             #tuner.scale_batch_size(model, datamodule=dm)

#             # if model_class == VAE_transformer_test_strided_conv:
#             #     model_name = "VAE_transformer_test_strided_conv"
#             # elif model_class == VAE_transformer_Parallel_test_strided_conv:
#             #     model_name = "VAE_transformer_Parallel_test_strided_conv"
#             # elif model_class == VAE_transformer:
#             #     model_name = "VAE_transformer_test_DecoderLength_hp"

#             trainer.fit(model, dm)
                                    
#             logging_file = "24_may_working_recon_error_hopefully_testFED_res.txt"
#             run_name = f"VAE_transformer_test_strided_conv_SwiGLU_lr_{lr}_beta_{beta}_gamma_{gamma}"
#             val_res = trainer.validate(
#                         dataloaders=dm.val_dataloader()
#                         )[0]
                        
# with open(os.path.join(logs_path, logging_file), "a") as f:
#         f.write(
#                 str(counter) + "\t" + "VAE_transformer_test_strided_conv" + "\t" + "SwiGLU" + "\t" +
#                 "lr: \t" + str(lr) + "\t" +
#                 "beta: \t" + str(beta) + "\t" +
#                 "gamma: \t" + str(gamma) + "\t" +
#                 "loss: \t" + format(val_res["validation_loss"], ".3f") + "\t" +        
#                 "recon_loss: \t" + format(val_res["validation_recon_loss"], ".3f") + "\t" +              
#                 "length_loss: \t" + format(val_res["validation_length_loss"], ".3f") + "\t" +
#                 "gamma*length_loss: \t" + format(val_res["validation_length_loss_gamma"], ".3f") + "\t" +
#                 "KL: \t" + format(val_res["validation_KL_loss"], ".3f") + "\t" +
#                 "beta*KL: \t" + format(val_res["validation_KL_beta"], ".3f")  + "\n"    
#                 )
# counter += 1
# stop_time = time.time()
# print("Time elapsed ", datetime.timedelta(seconds=stop_time-start_time))

# learning_rates = [5e-05]
# betas = [0.0001]
# gammas = [50]

# for model_class in models:
#     for activation in activations:
#         for lr in learning_rates:
#             for beta in betas:
#                 for gamma in gammas:
#                     # VAE or use VAE_transformer or use VAE_transformer_Parallel 
#                     model = model_class.load_from_checkpoint(
#                         checkpoint_path= "/home/scoutvdb/project/shared/scoutvdb/logs/12_may_hp_search_FED_val_set_logs/new_12_may_VAE_transformer_test_strided_conv_SwiGLU_lr_5e-05_beta_0.001_gamma_10/version_0/checkpoints/step_10000.ckpt",
#                         vocab_size = 33, 
#                         hidden_sizes = [16, 32, 64, 128, 256, 256], 
#                         bottleneck_size = 128, 
#                         learning_rate = lr, 
#                         blocks_per_stage = 2,
#                         n_heads = 8,
#                         n_warmup_steps = 1000,
#                         beta = beta,
#                         gamma=gamma,
#                         activation=activation,
#                         use_decoder_length=True
#                         )
                    
#                     if model_class == VAE_transformer_test_strided_conv:
#                         print(" \n Using VAE_transformer_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
#                     elif model_class == VAE_transformer_Parallel_test_strided_conv:
#                         print(" \n Using VAE_transformer_Parallel_test_strided_conv with " , activation, " in feedforward layer of Transformer \n")
#                     elif model_class == VAE_transformer:
#                         print(" \n Using VAE_transformer with " , activation, " in feedforward layer of Transformer \n")

#                     else:
#                         print("Something went wrong...")

#                     print(f"Using gamma as {gamma} and a batch size of {batch_size} beta {beta} lr {learning_rates}\n")

#                     callbacks = [
#                         ModelCheckpoint(monitor="validation_loss", mode="min", save_top_k=1, filename="best_val_loss_checkpoint"),  # Save the best checkpoint based on validation loss
#                         ModelCheckpoint(monitor=None, save_top_k=-1, every_n_train_steps=2500, filename="{step}"),
#                         ModelCheckpoint(monitor=None, save_top_k=1, every_n_train_steps=100, filename="checkpoint_{step}"), # Save a checkpoint every 100 steps, overwriting the previous one
#                         EarlyStopping(monitor='validation_loss', patience=7, verbose=True, mode='min')
#                     ]
#                     if model_class == VAE_transformer_test_strided_conv:
#                          name_logs = f"TEST_load_checkpoint_HPC_18_may_VAE_transformer_test_strided_conv_{activation}_lr_{lr}_beta_{beta}_gamma_{gamma}"
#                     elif model_class == VAE_transformer_Parallel_test_strided_conv:
#                          name_logs = f"TEST_new_VAE_transformer_Parallel_test_strided_conv_{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
#                     elif model_class == VAE_transformer:
#                          name_logs = f"TEST_new_VAE_transformer{activation} num_blocks_2 0_001 subsample lr_{lr}_beta_{beta}_gamma_{gamma}"
                         

#                     logger = TensorBoardLogger(
#                         logs_path, name=name_logs
#                     )

#                     trainer = pl.Trainer(
#                         max_steps=500,
#                         accelerator="gpu",
#                         logger = logger,
#                         val_check_interval=1000,
#                         check_val_every_n_epoch=None,
#                         devices=[0], # [1] This selects the second gpu on nvidia-smi
#                         callbacks=callbacks,
#                         gradient_clip_val=1 #gradient clipping on norm = 1
#                         )
                    
#                     #tuner = Tuner(trainer)
#                     #tuner.scale_batch_size(model, datamodule=dm)

#                     if model_class == VAE_transformer_test_strided_conv:
#                         model_name = "VAE_transformer_test_strided_conv"
#                     elif model_class == VAE_transformer_Parallel_test_strided_conv:
#                          model_name = "VAE_transformer_Parallel_test_strided_conv"
#                     elif model_class == VAE_transformer:
#                          model_name = "VAE_transformer_test_DecoderLength_hp"

#                     trainer.fit(model, dm)
                        
#                     logging_file = "test_loadcheckpoint_HPC_18_may_testFED_res.txt"
#                     run_name = f"{model_name} {activation} lr_{lr}_beta_{beta}_gamma_{gamma}"
#                     val_res = trainer.validate(
#                                 dataloaders=dm.val_dataloader()
#                                 )[0]
                        
#                     with open(os.path.join(logs_path, logging_file), "a") as f:
#                         f.write(
#                                 str(counter) + "\t" + str(model_name) + "\t" + str(activation) + "\t" +
#                                 "lr: \t" + str(lr) + "\t" +
#                                 "beta: \t" + str(beta) + "\t" +
#                                 "gamma: \t" + str(gamma) + "\t" +
#                                 "loss: \t" + format(val_res["validation_loss"], ".3f") + "\t" +        
#                                 "recon_loss: \t" + format(val_res["validation_recon_loss"], ".3f") + "\t" +              
#                                 "length_loss: \t" + format(val_res["validation_length_loss"], ".3f") + "\t" +
#                                 "gamma*length_loss: \t" + format(val_res["validation_length_loss_gamma"], ".3f") + "\t" +
#                                 "KL: \t" + format(val_res["validation_KL_loss"], ".3f") + "\t" +
#                                 "beta*KL: \t" + format(val_res["validation_KL_beta"], ".3f")  + "\n"    
#                                 )
#                     counter += 1
# stop_time = time.time()
# print("Time elapsed ", datetime.timedelta(seconds=stop_time-start_time))