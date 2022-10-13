import pytorch_lightning as pl
from DataModules import Cityscapes
from PSGAN import PSGAN
import torch
from options import Options

opt = Options().parse()

trainer = pl.Trainer(default_root_dir="./checkpoint-path",
		     accelerator="auto",
	             devices=1 if torch.cuda.is_available() else None,
                     max_epochs=opt.n_epochs)
dm = Cityscapes(split=opt.split)
model = PSGAN(lr=opt.lr, coef=opt.coef)
trainer.fit(model, dm)

# NASTAVI OD CHECKPOINTA
#trainer.fit(model, dm, ckpt_path='/home/ubuntu/Visage-Technologies-Internship/project/checkpoint-path/lightning_logs/version_22/checkpoints/epoch=472-step=1362240.ckpt')
#model = PSGAN.load_from_checkpoint('/home/ubuntu/Visage-Technologies-Internship/project/checkpoint-path/lightning_logs/version_23/checkpoints/epoch=999-step=2880000.ckpt')
trainer.test(model, dm)
# vizualizacija u tensorboardu
