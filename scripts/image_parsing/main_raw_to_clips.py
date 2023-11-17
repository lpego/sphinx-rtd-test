# %%
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature, measure, morphology, segmentation
from tqdm import tqdm

from mzbsuite.utils import cfg_to_arguments, noneparse


def main(args, cfg):
    """
    This script takes a folder of raw images and clips them into smaller images, with their mask.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments passed to the script. Namely:

            - input_dir: path to directory with raw images
            - output_dir: path to directory where to clip images
            - save_full_mask_dir: path to directory where to save labeled full masks
            - v (verbose): print more info
            - config_file: path to config file with per-script args

    cfg : argparse.Namespace
        Configuration with detailed parametrisations.

    Returns
    -------
    None. Everything is saved to disk.
    """

    PLOTS = False
    # define paths
    main_root = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.save_full_mask_dir is not None:
        args.save_full_mask_dir = Path(args.save_full_mask_dir)

    # get list of files to process
    files_proc = list(main_root.glob(f"**/*.{cfg.impa_image_format}"))
    # make sure weird capitalization doesn't cause issues
    files_proc.extend(list(main_root.glob(f"**/*.{cfg.impa_image_format.upper()}")))
    files_proc = [a for a in files_proc if "mask" not in str(a)]
    files_proc.sort()

    if args.verbose:
        print(f"parsing {len(files_proc)} files")

    # make sure that this will be general enough
    ### WE REALLY NEED TO CHANGE THIS!
    if "project_portable_flume" in str(main_root):
        location_cutout = [int(a) for a in cfg.impa_clip_areas]

    # define quick normalization function
    norm = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

    iterator = tqdm(files_proc, total=len(files_proc))
    for i, fo in enumerate(iterator):
        mask_props = []

        # get image path
        raw_image_in = fo
        full_path_raw_image_in = fo.resolve()

        # read image and convert to HSV
        img = cv2.imread(str(full_path_raw_image_in))[:, :, [2, 1, 0]]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        im_t = hsv[:, :, 0].copy()
        im_t = (255 * norm(np.mean(hsv[:, :, :2], axis=2))).astype(np.uint8)

        # filter image with some iterations of gaussian blur
        for _ in range(cfg.impa_gaussian_blur_passes):
            im_t = cv2.GaussianBlur(im_t, tuple(cfg.impa_gaussian_blur), 0)

        # prepare for morphological reconstruction
        seed = np.copy(im_t)
        seed[1:-1, 1:-1] = im_t.min()
        mask = np.copy(im_t)

        # remove the background
        dil = morphology.reconstruction(seed, im_t, method="dilation")
        im_t = (im_t - dil).astype(np.uint8)

        # adaptive local thresholding of foreground vs background
        # weighted cross correlation with gaussian filter
        ad_thresh = cv2.adaptiveThreshold(
            im_t,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            cfg.impa_adaptive_threshold_block_size,
            -2,
        )
        # additional global threhsold to remove foreground vs background
        t, thresh = cv2.threshold(im_t, 0, 255, cv2.THRESH_OTSU)

        # merge thresholds to globally get foreground masks
        # thresh = thresh | ad_thresh
        thresh = thresh + ad_thresh > 0

        # postprocess masking to remove small objects and fill holes
        kernel = np.ones(cfg.impa_mask_postprocess_kernel, np.uint8)
        for _ in range(cfg.impa_mask_postprocess_passes):
            thresh = cv2.morphologyEx(
                (255 * thresh).astype(np.uint8), cv2.MORPH_CLOSE, kernel
            )
            thresh = cv2.morphologyEx(
                (255 * thresh).astype(np.uint8), cv2.MORPH_OPEN, kernel
            )
        thresh = ndimage.binary_fill_holes(thresh)

        # cut out area related to measurement/color calibration widget
        ### WE REALLY NEED TO CHANGE THIS!
        if "project_portable_flume" in str(main_root):
            thresh[location_cutout[0] :, location_cutout[1] :] = 0

        # get labels of connected components
        labels = measure.label(thresh, connectivity=2, background=0)

        if PLOTS:
            f, a = plt.subplots(1, 4, figsize=(21, 9))
            a[0].imshow(thresh)
            a[1].imshow(ad_thresh)
            a[2].imshow(img)
            a[3].imshow(labels)
            plt.show()
            plt.savefig("test.png")

        # Save the labels as a jpg for the full image
        if args.save_full_mask_dir is not None:
            args.save_full_mask_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(args.save_full_mask_dir / f"labels_{fo.stem}.jpg").lower(),
                (labels).astype(np.uint8),
            )
            if not cfg.impa_save_clips_plus_features:
                if args.verbose:
                    print("skipping clip generation")
                continue

        # get region properties
        rprop = measure.regionprops(labels)
        mask = np.ones(thresh.shape, dtype="uint8")

        # init some stuff
        sub_df = pd.DataFrame([])
        c = 1
        # loop through identified regions and get some properties
        for label in range(len(rprop)):  # np.unique(labels):
            reg_pro = rprop[label]

            # skip background
            if reg_pro.label == 0:
                continue

            # skip small objects
            if reg_pro.area < cfg.impa_area_threshold:  # 5000 defauilt
                continue

            # get mask for current region of interest
            current_mask = np.zeros(thresh.shape)
            current_mask[labels == reg_pro.label] = 1

            # coordinates of bounding box corners for current region of interest
            (
                min_row,
                min_col,
                max_row,
                max_col,
            ) = reg_pro.bbox  # cv2.boundingRect(approx)
            (x, y, w, h) = (min_col, min_row, max_col - min_col, max_row - min_row)

            # get the bounding box with some buffer
            (x_e, y_e, w_e, h_e) = (
                np.max((x - cfg.impa_bounding_box_buffer, 0)),
                np.max((y - cfg.impa_bounding_box_buffer, 0)),
                w + 2 * cfg.impa_bounding_box_buffer,
                h + 2 * cfg.impa_bounding_box_buffer,
            )

            if PLOTS:
                f, a = plt.subplots(1, 1, figsize=(10, 6))
                a.imshow(img[:, :, [0, 1, 2]], aspect="auto")
                rect = plt.Rectangle(
                    (x_e, y_e), w_e, h_e, fc="none", ec="black", linewidth=2
                )
                a.add_patch(rect)
                plt.show()
                plt.savefig(f"test_mask{c}.png")
                exit()

            # get the crop of the image and the mask
            crop = img[y_e : y_e + h_e, x_e : x_e + w_e, [2, 1, 0]]
            crop_hsv = hsv[y_e : y_e + h_e, x_e : x_e + w_e, :]
            crop_mask = current_mask[y_e : y_e + h_e, x_e : x_e + w_e]
            crop_im_t = im_t[y_e : y_e + h_e, x_e : x_e + w_e]

            im_crop_m = crop.reshape(-1, 3)[
                crop_mask.reshape(
                    -1,
                ).astype(bool),
                :,
            ]
            hsv_crop_m = crop_hsv.reshape(-1, 3)[
                crop_mask.reshape(
                    -1,
                ).astype(bool),
                :,
            ]

            # save actual image and mask crops
            # Avoid "invalid value encountered in true_divide" warning
            np.seterr(divide="ignore", invalid="ignore")
            cv2.imwrite(
                str(outdir / (f"{fo.stem}_{c}_mask.{cfg.impa_image_format}").lower()),
                (255 * crop_mask / crop_mask).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

            # reactivate warnings
            np.seterr(divide="warn", invalid="warn")

            cv2.imwrite(
                str(outdir / (f"{fo.stem}_{c}_rgb.{cfg.impa_image_format}").lower()),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )
            # get average color of the crop
            # not really needed, aren't they
            # im_crop_cmean = str(np.mean(im_crop_m, axis=0))
            # hsv_crop_cmean = str(np.mean(hsv_crop_m, axis=0))

            # im_crop_std = str(np.std(im_crop_m, axis=0))
            # hsv_crop_std = str(np.std(hsv_crop_m, axis=0))

            mask = mask + current_mask * c

            if PLOTS:
                f, a = plt.subplots(1, 4, figsize=(10, 6))
                a[0].imshow(crop)
                a[1].imshow(reg_pro.image)  # crop_mask)
                a[2].imshow(
                    (
                        crop * np.transpose(np.tile(crop_mask, (3, 1, 1)), (1, 2, 0))
                    ).astype(np.uint8)
                )
                im_t_crop_m = crop_im_t.reshape(-1, 1)[
                    crop_mask.reshape(
                        -1,
                    ).astype(bool),
                    :,
                ]
                a[3].hist(im_t_crop_m, bins=50)

                plt.show()

            sub_df = {}
            sub_df["input_file"] = raw_image_in
            sub_df["species"] = raw_image_in.name.split(".")[0]
            sub_df["png_mask_id"] = c
            sub_df["reg_lab"] = reg_pro.label
            sub_df["squareness"] = w / float(h)
            # sub_df["average_color"] = im_crop_cmean
            # sub_df["average_color_std"] = im_crop_std
            # sub_df["average_hsv"] = hsv_crop_cmean
            # sub_df["average_hsv_std"] = hsv_crop_std
            sub_df["tight_bb"] = f"({x}, {y}, {w}, {h})"
            sub_df["large_bb"] = f"({x_e}, {y_e}, {w_e}, {h_e})"
            sub_df["ell_minor_axis"] = reg_pro.minor_axis_length
            sub_df["ell_major_axis"] = reg_pro.major_axis_length
            sub_df["bbox_area"] = reg_pro.bbox_area
            sub_df["area_px"] = reg_pro.area
            sub_df["mask_centroid"] = str(reg_pro.centroid)
            sub_df = pd.DataFrame(data=sub_df, index=[0])

            mask_props.append(sub_df)
            c += 1

    if not PLOTS:
        if mask_props:
            mask_props = pd.concat(mask_props).reset_index().drop(columns=["index"])
            mask_props.to_csv(outdir / "_mask_properties.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="path to config file with per-script args",
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="path to directory with raw images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory where to clip images",
    )
    parser.add_argument(
        "--save_full_mask_dir",
        type=str,
        required=False,
        default=None,
        help="path to directory where to save labeled full masks",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more info")
    args = parser.parse_args()

    print(args.config_file)

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    if args.verbose:
        print(f"main args: {args}")
        print(f"scripts config: {cfg}")

    sys.exit(main(args, cfg))
