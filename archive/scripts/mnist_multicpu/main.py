import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

BASE_PATH = (
    "/gws/ssde/j25a/ai4er/users/pn341/climate-rl/scripts/mnist_multicpu"
)
DATASET_PATH = "/gws/ssde/j25a/ai4er/users/pn341/datasets/mnist"
CPU_CORES = int(os.cpu_count()) - 1


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("valid_loss", loss, sync_dist=True)
        return {"loss": loss, "log": {"valid_loss": loss}}

    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("test_loss", loss, sync_dist=True)
        return {"loss": loss, "log": {"test_loss": loss}}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(
            MNIST(
                DATASET_PATH,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=64,
            num_workers=CPU_CORES,
        )

    def val_dataloader(self):
        return DataLoader(
            MNIST(
                DATASET_PATH,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=64,
            num_workers=CPU_CORES,
        )

    def test_dataloader(self):
        return DataLoader(
            MNIST(
                DATASET_PATH,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=64,
            num_workers=CPU_CORES,
        )


if __name__ == "__main__":
    model = MNISTModel()

    logger = loggers.TensorBoardLogger(
        f"{BASE_PATH}/logs/", name="mnist_model"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{BASE_PATH}/checkpoints/",
        filename="mnist-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="valid_loss",
        mode="min",
    )

    # Interactive GPU Host Training
    # trainer = Trainer(
    #     accelerator="gpu",
    #     max_epochs=5,
    #     deterministic=True,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    # )

    # Interactive CPU Host Training
    # trainer = Trainer(
    #     accelerator="cpu",
    #     max_epochs=5,
    #     deterministic=True,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    # )

    # SLURM Computing Cluster Multi-Node CPU Training
    trainer = Trainer(
        devices=1,
        accelerator="cpu",
        strategy="ddp",
        max_epochs=5,
        num_nodes=2,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)
    # trainer.test()

    model_path = f"{BASE_PATH}/checkpoints/mnist_final.ckpt"
    trainer.save_checkpoint(model_path)
    print(f"Model saved at {model_path}")
