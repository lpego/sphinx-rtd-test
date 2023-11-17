"""
Module containing utility functions for mzbsuite
C 2023, M. Volpi, Swiss Data Science Center
"""

from torchvision import models

from pathlib import Path

# import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    max_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
)


def noneparse(value):
    """
    Helper function to parse None values from YAML files

    Parameters
    ----------
    value: string
        string to be parsed

    Returns
    -------
    value: string or None
        parsed string
    """

    if value.lower() == "none":
        return None

    return value


class cfg_to_arguments(object):
    """
    This class is used to convert a dictionary to an object and extend the argparser.
    In the __init__ method, we iterate over the dictionary and add each key as an attribute to the object.
    Input is a dictionary, output is an object, that mimicks the argparse object.


    Example
    -------

        cfg = {'a': 1, 'b': 2}
    args = cfg_to_arguments(cfg)
    print(args.a) # 1
    print(args.b) # 2

    cfg can be from configs stored in YAML file, a JSON file, or a dictionary, whatever you prefer.
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        args: dict
            dictionary of arguments
        """
        for key in args:
            setattr(self, key, args[key])

    def __str__(self):
        """Prints the object as a string"""
        return self.__dict__.__str__()


def regression_report(y_true, y_pred, PRINT=True):
    """
    Helper function to print regression metrics. Taken and adapted from
    https://github.com/scikit-learn/scikit-learn/issues/18454#issue-708338254

    Parameters
    ----------
    y_true: np.array
        ground truth values
    y_pred: np.array
        predicted values
    PRINT: bool
        whether to print the metrics or not

    Returns
    -------
    metrics: list
        list of tuples with the name of the metric and its value
    """

    error = y_true - y_pred
    percentile = [5, 25, 50, 75, 95]
    percentile_value = np.percentile(error, percentile)

    metrics = [
        ("mean absolute error", mean_absolute_error(y_true, y_pred)),
        ("median absolute error", median_absolute_error(y_true, y_pred)),
        ("mean squared error", mean_squared_error(y_true, y_pred)),
        ("max error", max_error(y_true, y_pred)),
        ("r2 score", r2_score(y_true, y_pred)),
        ("explained variance score", explained_variance_score(y_true, y_pred)),
    ]

    if PRINT:
        print("Metrics for regression:")
        for metric_name, metric_value in metrics:
            print(f"{metric_name:>25s}: {metric_value: >20.3f}")

        print("\nPercentiles:")
        for p, pv in zip(percentile, percentile_value):
            print(f"{p: 25d}: {pv:>20.3f}")

    return metrics


def read_pretrained_model(architecture, n_class):
    """
    Helper script to load models compactly from pytorch model zoo and prepare them for Hummingbird finetuning

    Parameters
    ----------
    architecture: str
        name of the model to load
    n_class: int
        number of classes to finetune the model for

    Returns
    -------
    model : pytorch model
        model with last layer replaced with a linear layer with n_class outputs
    """

    architecture = architecture.lower()

    if architecture == "vgg":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        in_feat = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features=in_feat, out_features=n_class, bias=True
        )

        for param in model.features.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            if np.any([a == 2 for a in param.shape]):
                pass
            else:
                param.requires_grad = False

    elif architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        # for param in model.fc.parameters():
        #     param.requires_grad = True

    elif architecture == "densenet161":
        model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=n_class, bias=True
        )

        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet-b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet-b1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "vit16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.head.parameters():
            param.requires_grad = True

    elif architecture == "convnext-small":
        model = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        model.classifier[2] = nn.Linear(
            in_features=model.classifier[2].in_features, out_features=n_class, bias=True
        )
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[2].parameters():
            param.requires_grad = True
    else:
        raise OSError("Model not found")

    return model


def find_checkpoints(dirs=Path("lightning_logs"), version=None, log="val"):
    """
    Find the checkpoints for a given log

    Parameters
    ----------
    dirs: Path  (default: Path("lightning_logs"))
        path to the lightning_logs folder
    version: str (default: None)
        version of the log to use

    Returns
    -------
    chkp: str
        list of paths to checkpoints

    """

    if version:
        ch_sf = list(dirs.glob(f"{version}/checkpoints/*.ckpt"))
    else:  # pick last
        ch_sp = [a.parents[1] for a in dirs.glob("**/*.ckpt")]
        ch_sp.sort()
        ch_sf = list(ch_sp[-1].glob("**/*.ckpt"))

    chkp = [a for a in ch_sf if log in str(a.name)]

    return chkp


from pytorch_lightning.callbacks import Callback
from datetime import datetime
import yaml


class SaveLogCallback(Callback):
    """
    Callback to save the log of the training

    TODO: will need to be updated to save the log of the training in more detail and in a more
    structured way
    """

    def __init__(self, model_folder):
        # super().__init__()
        self.model_folder = model_folder

    # def on_train_start(self, trainer, pl_module):
    #     self.model_folder = self.model_folder / "checkpoints"
    #     # store locally some meta info, if file exists, append to it
    #     # this in each model folder
    #     flog = self.model_folder.parents[0] / "trn_date.yaml"
    #     flag = "a" if flog.is_file() else "w"
    #     with open(self.model_folder.parents[0] / "trn_date.yaml", flag) as f:
    #         yaml.safe_dump(
    #             {"train-date-start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f
    #         )

    #     # this is a global file containing all the training dates for all models
    #     flog = self.model_folder.parents[1] / "all_trn_date.yaml"
    #     flag = "a" if flog.is_file() else "w"
    #     with open(self.model_folder.parents[1] / "all_trn_date.yaml", flag) as f:
    #         yaml.safe_dump(
    #             {
    #                 f"{self.model_folder.parents[1].name}": {
    #                     "start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                 }
    #             },
    #             f,
    #         )

    def on_train_end(self, trainer, pl_module):
        """
        Save the end date of the training
        """

        # store locally some meta info, if file exists, append to it
        # this in each model folder
        flog = self.model_folder.parents[0] / "trn_date.yaml"
        flag = "a" if flog.is_file() else "w"
        with open(self.model_folder.parents[0] / "trn_date.yaml", flag) as f:
            yaml.safe_dump(
                {"train-date-end": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f
            )

        # this is a global file containing all the training dates for all models
        flog = self.model_folder.parents[1] / "all_trn_date.yaml"
        flag = "a" if flog.is_file() else "w"
        with open(self.model_folder.parents[1] / "all_trn_date.yaml", flag) as f:
            yaml.safe_dump(
                {
                    f"{self.model_folder.parents[0].name}": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                },
                f,
            )
