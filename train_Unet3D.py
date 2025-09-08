import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from segmentation.lit_Unet3D import LitUNet3D


def train_model(train_data, val_data, in_channels, out_channels, learning_rate=1e-3, batch_size=2, max_epochs=10):
    # Unpack the training and validation data
    raw_train, label_train = train_data
    raw_val, label_val = val_data

    # Create TensorDatasets
    train_dataset = TensorDataset(torch.tensor(raw_train, dtype=torch.float32), torch.tensor(label_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(raw_val, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = LitUNet3D(in_channels, out_channels, learning_rate)

    # Initialize a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=1 if torch.cuda.is_available() else 0)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    return model


