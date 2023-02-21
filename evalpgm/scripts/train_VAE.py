import sys
import pytorch_lightning as pl
from evalpgm.data import Uniref50DataModule
from evalpgm.models import VAE
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

data_path = str(sys.argv[1])
logs_path = str(sys.argv[2])

dm = Uniref50DataModule(data_path, batch_size = 256, n_workers = 8)

# model
model = VAE(
    vocab_size = 33, 
    hidden_sizes = [16, 32, 64, 128, 256], 
    bottleneck_size = 128, 
    learning_rate = 5e-4, 
    blocks_per_stage = 2,
    n_warmup_steps = 1000,
    beta = 0.001)

callbacks = [
    ModelCheckpoint(monitor="val_loss", mode="min"),
    EarlyStopping(monitor='val_loss', patience=7, verbose=True, mode='min')
    ]
logger = TensorBoardLogger(
    logs_path, name="use_this_name_to_specify_which_model_it_is_eg_hyperparameters"
)

trainer = pl.Trainer(
	max_steps=500_000,
	accelerator="gpu",
    logger = logger,
    val_check_interval=10_000,
    check_val_every_n_epoch=None,
	devices=1,
    callbacks=callbacks)

#tensorboard --logdir logs in terminal --> not showing up anything in browser
# NOTE TO SCOUT: params will be saved in the tensorboard log folder. The fact that it is not showing up in your browser is because tensorboard will be running on a port on the server, not on your machine.
# You can access tensorboard in vs code directly: Ctrl+Shift+P > "Launch Tensorboard"

#figure out where params etc are saved, good values for gradient clipping, LR scheduler, ...
# NOTE TO SCOUT: I added linear warmup (copy-paste from Lightning docs that I also use) to your optimization setup. In practice I feel that this should be enough.

trainer.fit(model, dm)
