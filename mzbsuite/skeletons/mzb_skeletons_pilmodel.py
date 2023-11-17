# %%
# import os, sys, time, copy
from pathlib import Path
from PIL import Image

# import datetime

import torch
import pytorch_lightning as pl

import numpy as np

# from torch import nn
# from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

# from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import F1Score

# from torchmetrics.functional import precision_recall
from torchvision import transforms

import segmentation_models_pytorch as smp

from mzbsuite.skeletons.mzb_skeletons_dataloader import MZBLoader_skels


# %%
class MZBModel_skels(pl.LightningModule):
    """
    Pytorch Lightning Module for training the skeleton segmentation model.

    Parameters
    ----------
    data_dir: str
        Path to the directory where the data is stored.
    pretrained_network: str
        Name of the pretrained network to use.
    learning_rate: float
        Learning rate for the optimizer.
    batch_size: int
        Batch size for the dataloader.
    weight_decay: float
        Weight decay for the optimizer.
    num_workers_loader: int
        Number of workers for the dataloader.
    step_size_decay: int
        Number of epochs after which the learning rate is decayed.
    num_classes: int
        Number of classes to predict.
    """

    def __init__(
        self,
        data_dir="data/skel_segm/",
        pretrained_network="efficientnet-b2",
        learning_rate=1e-4,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        step_size_decay=5,
        num_classes=2,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = Path(data_dir)
        self.learning_rate = learning_rate
        self.architecture = pretrained_network
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.step_size_decay = step_size_decay
        self.data_dir_tst = ""
        self.num_classes = num_classes

        # some written in stone stuff.
        self.im_folder = self.data_dir / "images"
        self.bo_folder = self.data_dir / "sk_body"
        self.he_folder = self.data_dir / "sk_head"

        # self.get_learnin_splits(self)

        np.random.seed(12)
        N = len(list(self.im_folder.glob("*.jpg")))
        self.trn_inds = sorted(
            list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
        )
        self.val_inds = sorted(list(set(np.arange(N)).difference(set(self.trn_inds))))

        self.size_im = 224
        self.dims = (3, self.size_im, self.size_im)

        # This defines data augmentation used for training
        self.transform_tr = transforms.Compose(
            [
                transforms.RandomRotation(degrees=[0, 360]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(
                    brightness=[0.8, 1.2], contrast=[0.8, 1.2]
                ),  # (brightness=[0.75, 1.25], contrast=[0.75, 1.25]), # was 0.8, 1.5
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # This defines data augmentation used for validation / testing
        self.transform_ts = transforms.Compose(
            [
                # transforms.CenterCrop((self.size_im, self.size_im)),
                transforms.Resize(
                    (self.size_im, self.size_im),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),  # AT LEAST 224
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # define the model, we use a Unet with a pretrained encoder, and a decoder with 2 output channels
        self.model = smp.Unet(
            encoder_name=self.architecture,
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation=None,
        )

        # Add maybe a torchmetrics F1 score?
        # self.f1 = F1Score(num_classes=2, average="micro")
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

        # set the loss function, here we use the dice loss / tversky extention,
        # which is more robust to class imbalance. one needs to set the alpha and beta hyperparameters
        self.loss_fn = smp.losses.TverskyLoss(
            smp.losses.MULTILABEL_MODE, alpha=0.2, beta=0.8
        )
        # usually, tversky loss is used with a focal loss, which is implemented in the following line
        # tversky loss hyperparameters are usually set to alpha=0.3, beta=0.7, or alpha=0.5, beta=0.5 for the focal loss
        self.save_hyperparameters()

    def set_learning_splits(self):
        """
        set the learning splits for training and validation
        """

        np.random.seed(12)
        N = len(list(self.im_folder.glob("*.jpg")))
        self.trn_inds = sorted(
            list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
        )
        self.val_inds = sorted(list(set(np.arange(N)).difference(set(self.trn_inds))))

        return self
        # channels, width, height = self.dims

    def forward(self, x):
        """
        forward pass of the model, returning logits
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        training iteration per batch
        """
        x, y, _ = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("trn_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        validation iteration per batch
        """
        x, y, _ = batch
        logits = self(x)  # [:, 1, ...]

        loss = self.loss_fn(logits, y)

        self.log(f"val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, print_log: str = "tst"):
        """
        test iteration per batch
        """
        # Reuse the validation_step for testing
        return self.validation_step(batch, batch_idx, print_log)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        """
        custom predict iteration per batch, returning probabilities and labels
        """
        x, y, _ = batch
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        return probs, y

    def configure_optimizers(self):
        """
        define the optimizer and the learning rate scheduler
        """
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # The scheduler needs a monitor over it. We use the validation loss.
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": self.step_size_decay
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    ######################
    # DATA RELATED HOOKS #
    ######################

    def train_dataloader(self, shuffle=True):
        """
        definition of train dataloader
        """
        trn_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="trn",
            ls_inds=self.trn_inds,
            transforms=self.transform_tr,
        )

        return DataLoader(
            trn_d,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.num_workers_loader,
        )

    def val_dataloader(self):
        """ "
        definition of custom val dataloader
        """
        val_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="val",
            ls_inds=self.val_inds,
            transforms=self.transform_ts,
        )

        return DataLoader(
            val_d,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )

    def train_ts_augm_dataloader(self):
        """
        def of a dataloader for training data using test-time data augmentation
        """
        trn_d = MZBLoader_skels(
            self.im_folder,
            self.bo_folder,
            self.he_folder,
            learning_set="trn",
            ls_inds=self.trn_inds,
            transforms=self.transform_ts,
        )

        return DataLoader(
            trn_d,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )

    def tst_dataloader(self):
        """
        def of custom test dataloader
        """
        return None

    def external_dataloader(self, data_dir):
        """
        def of custom test dataloader
        """
        dub_folder = Path(data_dir)

        tst_dube = MZBLoader_skels(
            dub_folder,
            Path(""),
            Path(""),
            learning_set="external",
            ls_inds=[],
            transforms=self.transform_ts,
        )

        return DataLoader(
            tst_dube,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers_loader,
        )
