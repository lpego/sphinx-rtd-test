import argparse
import os
import sys
import torch
import cv2

from datetime import datetime
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import thin
from torchvision import transforms
from tqdm import tqdm

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml

from mzbsuite.skeletons.mzb_skeletons_pilmodel import MZBModel_skels
from mzbsuite.skeletons.mzb_skeletons_helpers import paint_image_tensor, Denormalize
from mzbsuite.utils import cfg_to_arguments, find_checkpoints

# Set the thread layer used by MKL
os.environ["MKL_THREADING_LAYER"] = "GNU"


def main(args, cfg):
    """
    Function to run inference of skeletons (body, head) on macrozoobenthos images clips, using a trained model.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing the arguments passed to the script. Notably:

            - input_dir: path to the directory containing the images to be classified
            - input_type: type of input data, either "val" or "external"
            - input_model: path to the directory containing the model to be used for inference
            - output_dir: path to the directory where the results will be saved
            - save_masks: path to the directory where the masks will be saved
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

    model = MZBModel_skels()
    model.model = model.load_from_checkpoint(
        checkpoint_path=mod_path, map_location=torch.device("cpu")
    )

    model.data_dir = Path(args.input_dir)
    model.im_folder = model.data_dir / "images"
    model.bo_folder = model.data_dir / "sk_body"
    model.he_folder = model.data_dir / "sk_head"
    model.num_workers_loader = 4
    model.batch_size = 8

    # this is unfortunately necessary to get the model to work, reindex trn/val split
    np.random.seed(12)
    N = len(list(model.im_folder.glob("*.jpg")))
    model.trn_inds = sorted(
        list(np.random.choice(np.arange(N), size=int(0.8 * N), replace=False))
    )
    model.val_inds = sorted(list(set(np.arange(N)).difference(set(model.trn_inds))))
    model.eval()
    model.freeze()

    if args.input_type == "val":  # ("flume" in str(args.input_dir)) and
        dataloader = model.val_dataloader()
        dataset_name = "flume"
    elif args.input_type == "external":
        dataloader = model.external_dataloader(args.input_dir)
        dataset_name = "external"

    im_fi = dataloader.dataset.img_paths

    pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        precision=32,
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else 1,
        callbacks=[pbar_cb],
        enable_checkpointing=False,
        logger=False,
    )

    outs = trainer.predict(
        model=model, dataloaders=[dataloader], return_predictions=True
    )

    # aggregate predictions
    p = []
    gt = []
    for out in outs:
        p.append(out[0].numpy())
        gt.append(out[1].numpy())
    pc = np.concatenate(p)
    gc = np.concatenate(gt)

    # %%
    # nn body preds
    preds_size = []

    if args.verbose:
        print("Neural network predictions done, refining and saving skeletons...")

    for i, ti in tqdm(enumerate(im_fi), total=len(im_fi)):
        im = Image.open(ti).convert("RGB")

        # get original size of image for resizing predictions
        o_size = im.size

        # get predictions
        x = model.transform_ts(im)
        x = x[np.newaxis, ...]
        with torch.set_grad_enabled(False):
            p = torch.sigmoid(model(x)).cpu().numpy().squeeze()

        refined_skel = np.concatenate((p, np.zeros_like(p[0:1, ...])), axis=0)
        refined_skel = Image.fromarray(
            (255 * np.transpose(refined_skel, (1, 2, 0))).astype(np.uint8)
        )

        refined_skel = transforms.Resize(
            (o_size[1], o_size[0]),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )(refined_skel)
        refined_skel = np.transpose(np.asarray(refined_skel), (2, 0, 1))

        # mask out the edges of the image
        if (cfg.skel_label_buffer_on_preds > 0) and (not cfg.skel_label_clip_with_mask):
            mask = np.ones_like(x[0, 0, ...])
            mask[-cfg.skel_label_buffer_on_preds :, :] = 0
            mask[: cfg.skel_label_buffer_on_preds, :] = 0
            mask[:, : cfg.skel_label_buffer_on_preds] = 0
            mask[:, -cfg.skel_label_buffer_on_preds :] = 0

            mask = Image.fromarray(mask)
            mask = np.array(
                transforms.Resize(
                    (o_size[1], o_size[0]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )(mask)
            )
            refined_skel = [
                (thin(a) > 0).astype(float) * mask for a in refined_skel[0:2, ...] > 50
            ]
        elif cfg.skel_label_clip_with_mask:
            # load mask
            mask_insect = Image.open(
                cfg.glob_blobs_folder / ti.name[:-4] + "_mask.jpg"
            ).convert("RGB")
            mask_insect = np.array(mask_insect)[:, :, 0] > 0
            mask_insect = Image.fromarray(mask_insect)
            mask_insect = np.array(
                transforms.Resize(
                    (o_size[1], o_size[0]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )(mask_insect)
            )
            refined_skel = [
                (thin(a) > 0).astype(float) * mask_insect
                for a in refined_skel[0:2, ...] > 50
            ]

        else:
            # Refine the predicted skeleton image
            refined_skel = [
                (thin(a) > 0).astype(float) for a in refined_skel[0:2, ...] > 50
            ]

        refined_skel = [(255 * s).astype(np.uint8) for s in refined_skel]

        if args.save_masks:
            name = "_".join(ti.name.split("_")[:-1])
            cv2.imwrite(
                str(args.save_masks / f"{name}_body.jpg"),
                refined_skel[0],
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            cv2.imwrite(
                str(args.save_masks / f"{name}_head.jpg"),
                refined_skel[1],
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

        preds_size.append(
            pd.DataFrame(
                {
                    "clip_name": "_".join(ti.name.split(".")[0].split("_")[:-1]),
                    "nn_pred_body": [np.sum(refined_skel[0] > 0)],
                    "nn_pred_head": [np.sum(refined_skel[1] > 0)],
                }
            )
        )

    preds_size = pd.concat(preds_size)
    # out_dir = Path(
    #     f"{args.output_dir}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    # )
    out_dir = Path(f"{args.output_dir}")

    out_dir.mkdir(exist_ok=True, parents=True)

    preds_size.to_csv(out_dir / f"size_skel_supervised_model.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="path with images for inference",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        required=True,
        help="either 'val' or 'external'",
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="where to save skeleton measure predictions as csv",
    )
    parser.add_argument(
        "--save_masks",
        type=str,
        required=True,
        help="where to save skeleton masks predictions as jpg",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.save_masks is not None:
        args.save_masks = Path(f"{args.save_masks}")
        args.save_masks.mkdir(parents=True, exist_ok=True)

    args.output_dir = Path(args.output_dir)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    sys.exit(main(args, cfg))

    # %%
    # ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    # 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
    # 'resnext101_32x32d', 'resnext101_32x48d',
    # 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
    # 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
    # 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d',
    # 'se_resnext101_32x4d', 'densenet121',
    # 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4',
    # 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
    # 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2',
    # 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2',
    # 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5',
    # 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8',
    # 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1',
    # 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3',
    # 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d',
    # 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e',
    # 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d',
    # 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s',
    # 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s',
    # 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006',
    # 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040',
    # 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160',
    # 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006',
    # 'timm-regnety_008', 'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040',
    # 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160',
    # 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d',
    # 'timm-mobilenetv3_large_075', 'timm-mobilenetv3_large_100', 'timm-mobilenetv3_large_minimal_100',
    # 'timm-mobilenetv3_small_075', 'timm-mobilenetv3_small_100', 'timm-mobilenetv3_small_minimal_100',
    # 'timm-gernet_s', 'timm-gernet_m', 'timm-gernet_l', 'mit_b0', 'mit_b1', 'mit_b2',
    # 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2',
    # 'mobileone_s3', 'mobileone_s4']"
