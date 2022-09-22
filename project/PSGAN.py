import torch
import torch.nn as nn
import pytorch_lightning as pl
from a_projekt.networks import Generator, PersonDiscriminator, BackgroundDiscriminator
from a_projekt.data import Cityscapes

class PSGAN (pl.LightningModule):
    def __init__(self, lr=0.0002, batch_size=1):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.discriminatorP = PersonDiscriminator()
        self.discriminatorB = BackgroundDiscriminator()

        self.validation_z = torch.randn(10, self.hparams.latent_dim)

    def forward(self, x):
        return self.generator(x)
    
    def Person_loss(self, y_hat, y):
        return nn.MSELoss(y_hat, y)
    
    def Background_loss(self, y_hat, y):
        return nn.BCELoss(y_hat, y)

    def L1_loss(self, y_hat, y):
        return nn.L1Loss(y_hat, y)
    
    # sve popravi
    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        # !!!!! potencijalno maknemo uvjete, odmah treniramo i G i Dp i Db
        # train generator
        # train discriminatorP
        # train discriminatorB

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5   , 0.999))
        opt_dP = torch.optim.Adam(self.discriminatorP.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_dB = torch.optim.Adam(self.discriminatorB.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Nakon svakih 100 epoha smanji learning rate za 0.01
        # u originalu: lr_nova = lr_stara - 0.01 * lr_pocetna ----> majsličnije
        g_sch = torch.optim.lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.99)
        dP_sch = torch.optim.lr_scheduler.StepLR(opt_dP, step_size=100, gamma=0.99)
        dB_sch = torch.optim.lr_scheduler.StepLR(opt_dB, step_size=100, gamma=0.99)

        return [opt_g, opt_dP, opt_dB], [g_sch, dP_sch, dB_sch] # optimizers, lr_schedulers