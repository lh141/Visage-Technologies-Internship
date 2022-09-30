import torch, torchvision
import torch.nn as nn
import pytorch_lightning as pl
from networks import Generator, PersonDiscriminator, BackgroundDiscriminator
from options import opt


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: # ako je sloj konvolucijski
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1: # ako je sloj normalizacijski
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PSGAN (pl.LightningModule):
    def __init__(self, lr=opt.lr, batch_size=1, coef = opt.coef):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.discriminatorP = PersonDiscriminator()
        self.discriminatorB = BackgroundDiscriminator()
        
        self.generator.apply(init_weights)
        self.discriminatorB.apply(init_weights)
        self.discriminatorP.apply(init_weights)
    

    def Person_loss(self, y_hat, y):
        loss = nn.MSELoss()
        return loss(y_hat, y)
    
    def Background_loss(self, y_hat, y):
        loss = nn.BCELoss()
        return loss(y_hat, y)

    def L1_loss(self, y_hat, y):
        loss = nn.L1Loss()
        return loss(y_hat, y)
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch['real']
        noise = batch['noise']
        bbox = batch['bbox']
        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
        
        fake = self.generator(noise)
        real_pedestrian = real[:, :, y:y+h, x:x+w]
        fake_pedestrian = fake[:, :, y:y+h, x:x+w]
        noise_fake = torch.cat((noise, fake), 1) # stavimo sliku sa šumom i bez šuma jednu do druge
        noise_real = torch.cat((noise, real), 1)

        if optimizer_idx == 0: # train Gen
            pred_B_fake = self.discriminatorB(noise_fake)
            one = torch.ones_like(pred_B_fake)
            one = one.type_as(real) # stavimo na isti device (CPU, GPU)
            loss_background = self.Background_loss(pred_B_fake, one) # veći loss za generator ako diskriminator veli da je slika lažna
            
            loss_L1 = self.L1_loss(fake, real) * self.hparams.coef
            
            pred_P_fake = self.discriminatorP(fake_pedestrian)
            one = torch.ones_like(pred_P_fake)
            one = one.type_as(real) # stavimo na isti device (CPU, GPU)
            loss_person = self.Person_loss(pred_P_fake, one)
            
            G_loss = loss_person + loss_background + loss_L1

            self.log("G_loss", G_loss, prog_bar=True)

            return G_loss

        if optimizer_idx == 1: # train PersonDisc
            # lažni pješaci
            Pdisc_fake = self.discriminatorP(fake_pedestrian)
            zero = torch.ones_like(Pdisc_fake)
            zero = zero.type_as(real) # stavimo na isti device (CPU, GPU)
            fake_loss = self.Person_loss(Pdisc_fake, zero)
            # pravi pješaci
            Pdisc_real = self.discriminatorP(real_pedestrian)
            one = torch.ones_like(Pdisc_real)
            one = one.type_as(real)
            real_loss = self.Person_loss(Pdisc_real, one)

            Pdisc_loss = 0.5 * (fake_loss + real_loss)

            self.log("Pd_loss", Pdisc_loss, prog_bar=True)

            return Pdisc_loss

        if optimizer_idx == 2: # train BackDisc
            # lažni pješaci
            Bdisc_fake = self.discriminatorB(noise_fake.detach()) # ne treba gradijent računati za lažnu sliku
            zero = torch.ones_like(Bdisc_fake)
            zero = zero.type_as(real) # stavimo na isti device (CPU, GPU)
            fake_loss = self.Person_loss(Bdisc_fake, zero)
            # pravi pješaci
            Bdisc_real = self.discriminatorB(noise_real)
            one = torch.ones_like(Bdisc_real)
            one = one.type_as(real)
            real_loss = self.Person_loss(Bdisc_real, one)

            Bdisc_loss = 0.5 * (fake_loss + real_loss)

            self.log("Bd_loss", Bdisc_loss, prog_bar=True)

            return Bdisc_loss
            

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_dP = torch.optim.Adam(self.discriminatorP.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_dB = torch.optim.Adam(self.discriminatorB.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Nakon svakih 100 epoha smanji learning rate za 1%
        # u originalu: lr_nova = lr_stara - 0.01 * lr_pocetna ----> najsličnije
        g_sch = torch.optim.lr_scheduler.StepLR(opt_g, step_size=100, gamma=0.99)
        dP_sch = torch.optim.lr_scheduler.StepLR(opt_dP, step_size=100, gamma=0.99)
        dB_sch = torch.optim.lr_scheduler.StepLR(opt_dB, step_size=100, gamma=0.99)

        return [opt_g, opt_dP, opt_dB], [g_sch, dP_sch, dB_sch] # optimizers, lr_schedulers


    def test_step(self, batch, batch_idx):
        real, noise, bbox = batch
        x, y, w, h = bbox

        fake = self.generator(noise)
        real_pedestrian = real[:, y:h, x:w]
        fake_pedestrian = fake[:, y:h, x:w]
        self.real_fake = torch.cat((real, fake), 1)
        self.r_f_ped = torch.cat((real_pedestrian, fake_pedestrian), 1)

    
    def test_epoch_end(self):
        gridI = torchvision.utils.make_grid(self.real_fake)
        self.logger.experiment.add_image("real vs. fake image", gridI, self.current_epoch)

        gridP = torchvision.utils.make_grid(self.r_f_ped)
        self.logger.experiment.add_image("real vs. fake pedestrian", gridP, self.current_epoch)
