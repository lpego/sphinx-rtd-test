# %%
# import os, sys, time, copy
# import numpy as np

from pathlib import Path
from PIL import Image

# import datetime

import torch
import pytorch_lightning as pl

# from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics import F1Score

# from torchmetrics.utilities import check_forward_no_full_state

from torchvision import transforms

# try:
#     __IPYTHON__
# except:
#     prefix = ""  # or "../"
# else:
#     prefix = "../"  # or "../"

# sys.path.append(f"{prefix}src")

from mzbsuite.utils import read_pretrained_model

from mzbsuite.classification.mzb_classification_dataloader import (
    MZBLoader,
    Denormalize,
)


# %%
class MZBModel(pl.LightningModule):
    """
    pytorch lightning class definition and model setup

    Parameters
    ----------
    data_dir : str
        path to the directory containing the training and validation sets
    pretrained_network : str
        name of the pretrained network to use
    learning_rate : float
        learning rate for the optimizer
    batch_size : int
        batch size for the training and validation dataloaders
    weight_decay : float
        weight decay for the optimizer
    num_workers_loader : int
        number of workers for the dataloaders
    step_size_decay : int
        number of epochs after which the learning rate is decayed
    num_classes : int
        number of classes to classify
    """

    def __init__(
        self,
        data_dir="data/learning_sets/",
        pretrained_network="resnet50",
        learning_rate=1e-4,
        batch_size=32,
        weight_decay=1e-8,
        num_workers_loader=4,
        step_size_decay=5,
        num_classes=8,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.architecture = pretrained_network
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers_loader = num_workers_loader
        self.step_size_decay = step_size_decay
        self.data_dir_tst = None
        self.num_classes = num_classes
        # Hardcode some dataset specific attributes

        self.size_im = 224
        self.dims = (3, self.size_im, self.size_im)
        # channels, width, height = self.dims

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
                transforms.RandomResizedCrop(
                    (self.size_im, self.size_im), scale=(0.5, 1)
                ),
                # transforms.Resize(
                # (self.size_im, self.size_im), interpolation=transforms.InterpolationMode.BILINEAR
                # ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

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

        # Define PyTorch model
        self.model = read_pretrained_model(self.architecture, self.num_classes)

        # initialize F1Score metric
        result_metric = F1Score(task="multiclass", num_classes=num_classes)

        # # check if full_state_update=False can be used safely
        # safe_to_use = check_forward_no_full_state(result_metric)
        # if safe_to_use:
        #     result_metric = F1Score(full_state_update=False)
        self.accuracy = result_metric

        self.save_hyperparameters()

    def forward(self, x):
        """
        forward pass return unnormalised logits, normalise when needed
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        training iteration, per batch
        """
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # loss = F.nll_loss(logits, y, weight=torch.Tensor((1,2)).to("cuda"))
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("trn_loss", loss, prog_bar=True, sync_dist=True)
        self.log("trn_acc", self.accuracy, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, print_log: str = "val"):
        """
        validation iteration, per batch
        """
        x, y, _ = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # loss = F.nll_loss(logits, y, weight=torch.Tensor((1,2)).to("cuda"))
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log(f"{print_log}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{print_log}_acc", self.accuracy, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx, print_log: str = "tst"):
        """
        Test iteration, per batch. return validation function call
        """
        return self.validation_step(batch, batch_idx, print_log)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        x, y, _ = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs, y

    # def configure_optimizers(self):
    #     "optimiser config plus lr scheduler callback"
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay,
    #     )
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=self.step_size_decay, gamma=0.5
    #     )
    #     # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     # optimizer, milestones=[10, 20, 50], gamma=0.5
    #     # )

    #     return [optimizer], [lr_scheduler]

    # figure out how Plateau scheduler could work when val fits are too good.
    def configure_optimizers(self):
        """
        optimiser config plus lr scheduler callback
        """

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.step_size_decay,
                    cooldown=1,
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    ######################
    # DATA RELATED HOOKS #
    ######################
    def train_dataloader(self, shuffle=True):
        """
        training data loader
        """
        files = [
            folds
            for folds in sorted(list((self.data_dir / "trn_set").glob("*")))
            if "zz_" not in folds.name
        ]

        dir_dict_trn = {}
        for a in files:
            dir_dict_trn[a.name] = a

        trn_d = MZBLoader(
            dir_dict_trn, learning_set="trn", ls_inds=[], transforms=self.transform_tr
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            trn_d,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.num_workers_loader,
        )

        if not shuffle:
            return DataLoader(
                trn_d,
                batch_size=self.batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_workers_loader,
            )

        else:  # means weighted random sampling
            # make the number of draws comparable to a full sweep over a balanced set.
            # Should just train for longer, but "epochs" are now shorter
            num_samples = (trn_d.labels == 1).sum().item()

            weights = []
            for c in [0, 1]:
                weights.extend(
                    len(trn_d.labels[trn_d.labels == c])
                    * [len(trn_d.labels) / (trn_d.labels == c).sum()]
                )

            self.trn_weights = torch.Tensor(weights)
            sampler = WeightedRandomSampler(self.trn_weights, num_samples=num_samples)

            return DataLoader(
                trn_d,
                batch_size=self.batch_size,
                drop_last=True,
                num_workers=self.num_workers_loader,
                sampler=sampler,
            )

    def val_dataloader(self):
        """
        validation data loader
        """
        if "val_set" in self.data_dir.name:
            files = [
                folds
                for folds in sorted(list((self.data_dir).glob("*")))
                if "zz_" not in folds.name
            ]
        else:
            files = [
                folds
                for folds in sorted(list((self.data_dir / "val_set").glob("*")))
                if "zz_" not in folds.name
            ]

        dir_dict_val = {}
        for a in files:
            dir_dict_val[a.name] = a

        val_d = MZBLoader(
            dir_dict_val, learning_set="val", ls_inds=[], transforms=self.transform_ts
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            val_d,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers_loader,
        )

    # def mixed_dataloader(self):
    #     dir_dict_tst = {}
    #     dir_dict_tst["zz_mixed"] = Path(self.data_dir_tst)

    #     tst_d = MZBLoader(
    #         dir_dict_tst,
    #         learning_set="tst",
    #         ls_inds=[],
    #         transforms= self.transform_ts
    #         )

    #     # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
    #     return DataLoader(
    #             tst_d,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers_loader,
    #         )

    # def tst_dataloader(self):
    #     "def of custom test dataloader"
    #     return None

    def external_dataloader(self, data_dir, glob_pattern="*_rgb.*"):
        """
        external data loader
        """
        dir_dict_tst = {}
        dir_dict_tst["unlab"] = Path(data_dir)

        tst_d = MZBLoader(
            dir_dict_tst,
            learning_set="tst",
            ls_inds=[],
            glob_pattern=glob_pattern,
            transforms=self.transform_ts,
        )

        # number of draws from the weighted random samples matches the 2 * (n_positive // batch_size)
        return DataLoader(
            tst_d,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers_loader,
        )
