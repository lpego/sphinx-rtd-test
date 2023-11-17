# %% test skimage skeletonize
import copy
import sys
from pathlib import Path
from datetime import datetime
import argparse

import cv2
import yaml

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, medial_axis, thin
from tqdm import tqdm

from mzbsuite.skeletons.mzb_skeletons_helpers import (
    get_endpoints,
    get_intersections,
    paint_image,
    segment_skel,
    traverse_graph,
)

from mzbsuite.utils import cfg_to_arguments, noneparse


def main(args, cfg):
    """
    Main function for skeleton estimation (body size) in the unsupervised setting.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from command line. Namely:
        
            - config_file: path to the configuration file
            - input_dir: path to the directory containing the masks
            - output_dir: path to the directory where to save the results
            - save_masks: path to the directory where to save the masks as jpg
            - list_of_files: path to the csv file containing the classification predictions
            - v (verbose): whether to print more info
            
    cfg : argparse.Namespace
        Arguments parsed from the configuration file.

    Returns
    -------
    None. All is saved to disk at specified locations.
    """
    PLOTS = False

    if args.save_masks is not None:
        args.save_masks = Path(f"{args.save_masks}")
        args.save_masks.mkdir(parents=True, exist_ok=True)

    # setup some area-specific parameters for filtering
    area_class = {
        0: {"area": [0, 10000], "thinning": 1, "lmode": "skeleton"},
        2: {"area": [10000, 15000], "thinning": 9, "lmode": "skeleton"},
        3: {"area": [15000, 20000], "thinning": 11, "lmode": "skeleton"},
        4: {"area": [20000, 50000], "thinning": 11, "lmode": "skeleton"},
        5: {"area": [50000, 100000], "thinning": 15, "lmode": "skeleton"},
        6: {"area": [100000, np.inf], "thinning": 20, "lmode": "skeleton"},
    }

    # Load in all masks in the input directory
    mask_list = sorted(
        list(Path(args.input_dir).glob(f"*_mask.{cfg.impa_image_format}"))
    )

    if args.list_of_files is not None:
        # select all files that are not predicted as "error" by the classification model
        predictions = (
            pd.read_csv(args.list_of_files).set_index("file").sort_values("file")
        )
        exclude = predictions[
            predictions[cfg.skel_class_exclude] > 1.0 / cfg.infe_num_classes
        ].index.to_list()
        exclude = [
            ("_".join(a.split("_")[:-1]) + f"_mask.{cfg.impa_image_format}").lower()
            for a in exclude
        ]
    else:
        exclude = []

    # load in file names that are classified as error by our CNN
    err_filenames = sorted(
        list(
            Path(
                f"{cfg.glob_root_folder}/data/learning_sets/project_portable_flume/curated_learning_sets/errors"
            ).glob("*.png")
        )
    )
    exclude += [
        ("_".join(a.name.split("_")[:-1]) + f"_mask.{cfg.impa_image_format}").lower()
        for a in err_filenames
    ]

    files_to_skel = [a for a in mask_list if a.name.lower() not in exclude]

    # %%
    out_dir = (
        args.output_dir
        / f"{args.input_dir.name}_unsupervised_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # %%
    growing_df = []
    # Load the image
    # PLOTS = True

    iterator = tqdm(files_to_skel, total=len(files_to_skel))
    # iterator = tqdm([args.input_dir / "1_ob_mixed_difficutly_clip_32_mask.jpg"])
    for fo in iterator:
        iterator.set_description(fo.name)
        # read in mask and rgb, rgb only for plotting
        mask_ = (cv2.imread(str(fo))[:, :, 0] / 255).astype(float)

        # Get needed filter size based on area
        for aa in area_class:
            if area_class[aa]["area"][0] < np.sum(mask_) < area_class[aa]["area"][1]:
                dpar = area_class[aa]["thinning"]

        # Find the medial axis, threshold it and clean if multiple regions, keep largest
        _, distance = medial_axis(mask_, return_distance=True)
        mask_dist = distance > dpar
        regs = label(mask_dist)
        props = regionprops(regs)

        # keep only the largest region of the eroded mask
        mask = regs == np.argmax([p.area for p in props if p.label > 0]) + 1

        # compute general skeleton by thinning the maks
        skeleton = thin(mask, max_num_iter=None)

        # get coordinates of point that intersect or are ends of the skeleton segments
        inter = get_intersections(skeleton=skeleton.astype(np.uint8))
        endpo = get_endpoints(skeleton=skeleton.astype(np.uint8))

        if args.save_masks:
            # save the skeletonized mask
            cv2.imwrite(
                str(args.save_masks / f"{''.join(fo.name.split('.')[:-1])}_skel.jpg"),
                (255 * skeleton / np.max(skeleton)).astype(np.uint8),
            )

        if PLOTS:
            rgb_ = cv2.imread(str(fo)[:-8] + "rgb.jpg")[:, :, [2, 1, 0]].astype(
                np.uint8
            )
            rgb_fi = paint_image(rgb_, skeleton, color=[255, 0, 0])
            rgb_ma = paint_image(rgb_, mask, color=[255, 0, 255])

        if inter:
            # then, deduplicate the intersections
            skel_labels, edge_attributes, skprop = segment_skel(skeleton, inter, conn=1)
            ds = distance_matrix(inter, inter) + 100 * np.eye(len(inter))
            duplicates = np.where(ds < 3)[0]
            try:
                inter = [a for a in inter if a != inter[duplicates[0]]]
            except:
                pass
        else:
            skel_labels = []

        # case for which there are no segments (ie, only one)
        if len(np.unique(skel_labels)) < 3:
            sub_df = pd.DataFrame(
                data={
                    "clip_filename": fo.name,
                    "conv_rate_mm_px": [cfg.skel_conv_rate],
                    "skel_length": [np.sum(skeleton)],
                    "skel_length_mm": [np.sum(skeleton) / cfg.skel_conv_rate],
                    "segms": [[0]],
                    "area": np.sum(mask_),
                }
            )
            growing_df.append(sub_df)

            if PLOTS:
                f, a = plt.subplots(1, 2)
                a[0].imshow(rgb_fi)
                a[1].imshow(rgb_ma)
                plt.title(f"Area: {np.sum(mask_)}")

        else:
            # remove nodes that are too close (less than 3px) and treat them as only one node
            # skel_labels, edge_attributes, skprop = segment_skel(skeleton, inter, conn=1)
            # ds = distance_matrix(inter, inter) + 100 * np.eye(len(inter))
            # duplicates = np.where(ds < 3)[0]
            # try:
            #     inter = [a for a in inter if a != inter[duplicates[0]]]
            # except:
            #     pass

            if args.save_masks:
                skel_masks_path = Path(args.save_masks)
                skel_masks_path.mkdir(parents=True, exist_ok=True)
                # save the skeletonized mask
                cv2.imwrite(
                    str(
                        args.save_masks / f"{''.join(fo.name.split('.')[:-1])}_skel.jpg"
                    ),
                    (255 * skel_labels / np.max(skel_labels)).astype(np.uint8),
                )

            # get the segments that touch each intersection, and make them neighbors
            intersection_nodes = []
            for coord in inter:
                local_cut = skel_labels[
                    (coord[1] - 4) : (coord[1] + 5), (coord[0] - 4) : (coord[0] + 5)
                ]
                nodes_touch = np.unique(local_cut[local_cut != 0])
                intersection_nodes.append(list(nodes_touch))

            # remove duplicates
            k = sorted(intersection_nodes)
            dedup = [k[i] for i in range(len(k)) if i == 0 or k[i] != k[i - 1]]
            intersection_nodes = dedup

            # get the segments that touch each endpoint
            dead_ends = []
            for coord in endpo:
                ends = skel_labels[
                    (coord[1] - 4) : (coord[1] + 5), (coord[0] - 4) : (coord[0] + 5)
                ]
                end_node = np.unique(ends[ends != 0])
                dead_ends.append(list(end_node))
            dead_ends = sorted(dead_ends)

            # build the graph of segments of the skeleton
            graph = {}
            for nod in np.unique(skel_labels[skel_labels > 0]):
                nei = [a for a in intersection_nodes if nod in a]
                nei = [item for sublist in nei for item in sublist]
                graph[nod] = list(set(nei).difference([nod]))

            end_nodes = copy.deepcopy(dead_ends)

            # tf is this
            end_nodes = [i for a in end_nodes for i in a]
            all_paths = []
            c = 0

            # traverse the graph for all end_nodes and get paths, append them to all_paths
            for init in end_nodes[:1]:
                p_i = traverse_graph(graph, init, end_nodes, debug=False)
                all_paths.extend(p_i)

            # remove doubles
            skel_cand = []
            for sk in all_paths:
                if sorted(sk) not in skel_cand:
                    skel_cand.append(sorted(sk))

            # measure path lenghts and keep max one, that is the skel for you
            sk_l = []
            for sk in skel_cand:
                cus = 0
                for i in sk:
                    cus += edge_attributes[i]
                sk_l.append(cus)

            # append to dataframe, some propeties
            sub_df = pd.DataFrame(
                data={
                    "clip_filename": fo.name,
                    "conv_rate_mm_px": [cfg.skel_conv_rate],
                    "skel_length": [sk_l[np.argmax(sk_l)]],
                    "skel_length_mm": [sk_l[np.argmax(sk_l)] / cfg.skel_conv_rate],
                    "segms": [skel_cand[np.argmax(sk_l)]],
                    "area": np.sum(mask_),
                }
            )
            growing_df.append(sub_df)

            if PLOTS:
                f, a = plt.subplots(1, 3, figsize=(12, 12))
                a[0].imshow(
                    paint_image(
                        skel_labels * 255,
                        dilation(skel_labels > 0, disk(3)),
                        [255, 0, 255],
                    )
                )

                a[0].scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
                a[0].scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
                for i in np.unique(skel_labels[skel_labels > 0]):
                    a[0].text(
                        x=skprop[i - 1].centroid[1],
                        y=skprop[i - 1].centroid[0],
                        s=f"{i}",
                        color="white",
                    )

                sel_skel = np.zeros_like(skel_labels)
                for i in np.unique(skel_labels[skel_labels > 0]):
                    if i in skel_cand[np.argmax(sk_l)]:
                        sel_skel += dilation(skel_labels == i, disk(3))
                sel_skel = sel_skel > 0

                a[1].imshow(paint_image(rgb_fi, sel_skel, [255, 0, 0]))
                a[2].imshow(rgb_ma)

                a[0].title.set_text(f"Area: {np.sum(mask_)}")
                a[1].title.set_text(f"Sel Segm: {skel_cand[np.argmax(sk_l)]}")
                a[2].title.set_text(f"Skel_lenght_px {sk_l[np.argmax(sk_l)]}")

    full_df = pd.concat(growing_df)
    full_df.to_csv(out_dir / "skeleton_attributes.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--list_of_files", type=noneparse, required=False, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_masks", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = cfg_to_arguments(cfg)

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    sys.exit(main(args, cfg))

# %% some visualizations for debugging
if 0:
    rgb_ = cv2.imread(str(fo)[:-8] + "rgb.png")[:, :, [2, 1, 0]].astype(np.uint8)
    rgb_fi = paint_image(rgb_, skeleton, color=[255, 0, 0])
    rgb_ma = paint_image(rgb_, mask, color=[255, 0, 255])

    labs = np.unique(skel_labels[skel_labels > 0])
    for i in labs:
        plt.figure()
        plt.imshow(
            paint_image(
                rgb_, dilation(skel_labels == i, disk(3)), [i / len(labs) * 255, 0, 255]
            )
        )
        plt.title(i)

    plt.figure()
    # plt.imshow(paint_image(rgb_, dilation(skel_labels, disk(3)), [255, 0, 255]))
    plt.imshow(dilation(skel_labels, disk(3)))
    plt.scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
    plt.scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
    for i in np.unique(skel_labels[skel_labels > 0]):
        plt.text(
            x=skprop[i - 1].centroid[1],
            y=skprop[i - 1].centroid[0],
            s=f"{i}",
            color="white",
        )

    f, a = plt.subplots(1, 3, figsize=(12, 12))
    a[0].imshow(
        paint_image(
            skel_labels * 255, dilation(skel_labels > 0, disk(3)), [255, 0, 255]
        )
    )

    a[0].scatter(np.array(inter)[:, 0], np.array(inter)[:, 1])
    a[0].scatter(np.array(endpo)[:, 0], np.array(endpo)[:, 1], marker="s")
    for i in np.unique(skel_labels[skel_labels > 0]):
        a[0].text(
            x=skprop[i - 1].centroid[1],
            y=skprop[i - 1].centroid[0],
            s=f"{i}",
            color="white",
        )

    sel_skel = np.zeros_like(skel_labels)
    for i in np.unique(skel_labels[skel_labels > 0]):
        # if i in skel_cand[np.argmax(sk_l)]:
        sel_skel += dilation(skel_labels == i, disk(3))
    sel_skel = sel_skel > 0

    a[1].imshow(paint_image(rgb_fi, sel_skel, [255, 0, 0]))
    a[2].imshow(rgb_ma)

    a[0].title.set_text(f"Area: {np.sum(mask_)}")
    a[1].title.set_text(f"Sel Segm: {skel_cand[np.argmax(sk_l)]}")
    a[2].title.set_text(f"Skel_lenght_px {sk_l[np.argmax(sk_l)]}")
