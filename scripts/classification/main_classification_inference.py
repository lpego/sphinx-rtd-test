import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
import torch

from mzbsuite.classification.mzb_classification_pilmodel import MZBModel
from mzbsuite.utils import cfg_to_arguments, find_checkpoints

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


def main(args, cfg):
    """
    Function to run inference on macrozoobenthos images clips, using a trained model.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script. Notably:

            - input_dir: path to the directory containing the images to be classified
            - input_model: path to the directory containing the model to be used for inference
            - output_dir: path to the directory where the results will be saved
            - config_file: path to the config file with train / inference parameters

    cfg : dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the results in the specified folder.
    """

    torch.hub.set_dir("./models/hub/")

    dirs = find_checkpoints(
        Path(args.input_model).parents[0],
        version=Path(args.input_model).name,
        log=cfg.infe_model_ckpt,
    )

    mod_path = dirs[0]
    # print(mod_path)

    model = MZBModel(
        pretrained_network=cfg.trcl_model_pretrarch,
    )

    model = model.load_from_checkpoint(checkpoint_path=mod_path, map_location="cpu")
    # torch.device("gpu")
    #   if torch.cuda.is_available()
    #  else torch.device("cpu"),
    # )

    model.to("cpu")
    model.data_dir = Path(args.input_dir)
    model.num_classes = cfg.infe_num_classes
    model.num_workers_loader = 4
    model.batch_size = 8
    model.eval()

    if "val_set" in model.data_dir.name:
        dataloader = model.val_dataloader()
    else:
        dataloader = model.external_dataloader(
            model.data_dir, glob_pattern=cfg.infe_image_glob
        )

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
        callbacks=[pbar_cb],
        enable_checkpointing=False,
        logger=False,
    )

    outs = trainer.predict(
        model=model, dataloaders=[dataloader]  # , return_predictions=True
    )

    if cfg.lset_taxonomy:
        mzb_taxonomy = pd.read_csv(Path(cfg.lset_taxonomy))
        mzb_taxonomy = mzb_taxonomy.drop(columns=["Unnamed: 0"])
        mzb_taxonomy = mzb_taxonomy.ffill(axis=1)
        # watch out this sorted is important for the class names to be in the right order
        class_names = sorted(
            list(mzb_taxonomy[cfg.lset_class_cut].str.lower().unique())
        )

    y = []
    p = []
    gt = []

    for out in outs:
        y.append(out[0].numpy().squeeze())
        p.append(out[1].numpy().squeeze())
        gt.append(out[2].numpy().squeeze())
    try:
        yc = np.concatenate(y)
        pc = np.concatenate(p)
        gc = np.concatenate(gt)
    except:
        yc = np.array(y)
        pc = np.asarray(p)
        gc = np.asarray(gt)

    # make now output csv containing the file name, the class and the probabilities of prediction.
    # if available, also add the ground truth class
    data = {
        "file": [f.name for f in dataloader.dataset.img_paths],
        "pred": np.argmax(pc, axis=1),
    }

    for clanam in class_names:
        data[clanam] = pc[:, class_names.index(clanam)]

    if "val_set" in model.data_dir.name:
        data["gt"] = gc
    else:
        data["gt"] = 0

    # out_dir = (
    #     Path(args.output_dir)
    #     / f"{model.data_dir.name}_{Path(args.input_model).name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    # )
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    csv_name = f"predictions.csv"
    df_ = pd.DataFrame(data=data)
    df_.to_csv(out_dir / csv_name, index=False)

    # %%
    if "val_set" in model.data_dir.name:
        from matplotlib import pyplot as plt
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            classification_report,
        )

        # cmat = confusion_matrix(gc, np.argmax(pc, axis=1), normalize="true")
        f = plt.figure(figsize=(10, 10))
        axis = f.gca()
        cm_disp = ConfusionMatrixDisplay.from_predictions(
            gc,
            yc,
            ax=axis,
            values_format=".1f",
            normalize=None,
            xticks_rotation="vertical",
            cmap="Greys",
            display_labels=class_names,
        )
        plt.savefig(out_dir / f"confusion_matrix.{cfg.glob_local_format}", dpi=300)

        rep_txt = classification_report(
            gc, np.argmax(pc, axis=1), target_names=class_names, zero_division=0
        )
        with open(out_dir / "classification_report.txt", "w") as f:
            f.write(rep_txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file with per-script args",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path with images to perform inference on",
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="path to model checkpoint for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to where to save classificaiton predictions as csv",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    sys.exit(main(args, cfg))
