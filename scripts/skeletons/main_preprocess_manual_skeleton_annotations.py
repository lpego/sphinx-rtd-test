# %%
# This script is not working as the filename have been changing in the new pipeline derived files.
# The actual skeleton learning sets are stable, as those were copied over, but we might think about redoing this script in the future.

import sys
import shutil
import argparse
import yaml
import json

import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from matplotlib import pyplot as plt

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True

sys.path.append(f"{prefix}")

from mzbsuite.utils import cfg_to_arguments  # , noneparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--input_raw_dir", type=str, required=True)
parser.add_argument("--input_clips_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()

# args = {}
# args["config_file"] = f"{prefix}configs/global_configuration.yaml"
# args[
#     "input_raw_dir"
# ] = f"{prefix}data/raw/2021_swiss_invertebrates/manual_measurements/"
# args["input_clips_dir"] = f"{prefix}/data/derived/project_portable_flume/blobs/"
# args[
#     "output_dir"
# ] = f"{prefix}data/learning_sets/project_portable_flume/skeletonization/"
# args["verbose"] = True
# args = cfg_to_arguments(args)

with open(args.config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg = cfg_to_arguments(cfg)

args.input_raw_dir = Path(args.input_raw_dir)
args.input_clips_dir = Path(args.input_clips_dir)
args.output_dir = Path(args.output_dir)

# if any of the folders exist, interrupt the script and raise en error.
if (args.output_dir).exists() and (
    (args.output_dir / "images")
    or (args.output_dir / "sk_body")
    or (args.output_dir / "sk_head")
):
    # print in red and then back to normal color
    raise ValueError(
        f"\033[91m{args.output_dir} already exists and contains data. Please delete or sprecify another folder.\033[0m"
    )
else:
    args.output_dir.mkdir(exist_ok=True, parents=True)

cfg.skel_save_attributes = Path(cfg.skel_save_attributes)
cfg.skel_save_attributes.mkdir(exist_ok=True, parents=True)
# %%
# define empty lists to store the data, columns and files to read
measures = []
cols = ["species", "clip_name", "head", "body"]
files_to_merge = list(sorted(args.input_raw_dir.glob("**/*/*.json")))

# loop over the files and extract the data, then append to the list, then merge, then save.
for jfi in files_to_merge[:]:
    clip_name = jfi.parent.name.split("__")[-1]
    species = clip_name.split("_")[2]

    with open(jfi) as f:
        data = json.load(f)

    body = data["line"]["body"]["data"]["lengths"]
    head = data["line"]["head"]["data"]["lengths"]

    measures.append(
        pd.DataFrame(
            {
                "clip_name": clip_name,
                "species": species,
                "head_length": head,
                "body_length": body,
            }
        )
    )

all_measures = pd.concat(measures)
all_measures.to_csv(
    cfg.skel_save_attributes / "manual_annotations_summary.csv", index=False
)
# %%
# Get clip based on the fact that it is an existing annotation in Daninas folder
annot_files = sorted(list(args.input_raw_dir.glob("**/*/line_V4.csv")))

(args.output_dir / "images").mkdir(exist_ok=True, parents=True)
(args.output_dir / "sk_body").mkdir(exist_ok=True, parents=True)
(args.output_dir / "sk_head").mkdir(exist_ok=True, parents=True)

# %%
# Loop over the annotations and copy the image and save the manual skeleton.
for file in annot_files[:1]:
    gen_name = "_".join(file.parent.name.split("__")[1].split("_")[:-1])
    rgb_clip = gen_name + f"_rgb.{cfg.impa_image_format}"

    # Copy the image to the output folder
    shutil.copy(args.input_clips_dir / rgb_clip, args.output_dir / "images")

    # Read the image and the annotation, to get the size of the image
    test_f_im = Path(args.input_clips_dir / rgb_clip)
    test_im = cv2.cvtColor(cv2.imread(str(test_f_im)), cv2.COLOR_BGR2RGB)

    if PLOTS:
        plt.figure()
        plt.imshow(test_im)

    # Read the annotation file and get the head and body line coordinates
    line = pd.read_csv(file)
    body = line[["x_coords", "y_coords"]].loc[line.annotation_id == "body"].values
    head = line[["x_coords", "y_coords"]].loc[line.annotation_id == "head"].values

    # Draw the lines corresponding to body size on the image and save
    bw_mask = np.zeros_like(test_im)
    body = cv2.polylines(
        bw_mask,
        [body],
        isClosed=False,
        color=(255, 255, 255),
        thickness=cfg.skel_label_thickness,
    )
    cv2.imwrite(str(args.output_dir / "sk_body" / f"{gen_name}_body_skel.png"), body)

    if PLOTS:
        plt.figure()
        plt.imshow(body)

    # Draw the lines corresponding to head size on the image and save
    bw_mask = np.zeros_like(test_im)
    head = cv2.polylines(
        bw_mask,
        [head],
        isClosed=False,
        color=(255, 255, 255),
        thickness=cfg.skel_label_thickness,
    )
    cv2.imwrite(str(args.output_dir / "sk_head" / f"{gen_name}_head_skel.png"), head)

    if PLOTS:
        plt.figure()
        plt.imshow(head)
