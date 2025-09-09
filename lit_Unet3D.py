import torch 
import torch.nn as nn
import pytorch_lightning as pl
# import the 3d unet model from 3dunet.py
from segmentation.models.Unet3D import UNet3D  


class LitUNet3D(pl.LightningModule):
    def __init__(self, in_channels, out_channels, learning_rate=1e-3):
        super(LitUNet3D, self).__init__()
        self.save_hyperparameters()
        self.model = UNet3D(in_channels, out_channels)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()  # Example loss function

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer