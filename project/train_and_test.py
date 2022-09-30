import pytorch_lightning as pl
from DataModules import Cityscapes
from PSGAN import PSGAN


trainer = pl.Trainer(max_epochs=200)
dm = Cityscapes()
model = PSGAN()
trainer.fit(model, dm)

trainer.test(model)
# vizualizacija u tensorboardu