# Module containing helper functions for the skeletonization scripts.

from pathlib import Path
from typing import List, Tuple, Union
import torch

import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


# %%
def paint_image(image: np.ndarray, mask: np.array, color: List[float]) -> np.ndarray:
    """
    Given an input image, a binary mask indicating where to paint, and a color to use,
    returns a new image where the pixels within the mask are colored with the specified color.

    Parameters
    ----------
        image (np.ndarray): Input image to paint.
        mask (np.array): Binary mask indicating where to paint.
        color (List[float]): List of 3 floats representing the RGB color to use.

    Returns
    -------
        rgb_fi (np.ndarray): New image with painted-in mask.
    """

    # Color the pixels within the mask with the specified color
    rgb_fi = image.copy()
    if len(rgb_fi.shape) == 2:
        rgb_fi = rgb_fi[:, :, np.newaxis]
        rgb_fi = np.concatenate((rgb_fi, rgb_fi, rgb_fi), axis=2)

    if np.max(rgb_fi) <= 1:
        rgb_fi = (rgb_fi * 255).astype(np.uint8)

    # Color the pixels within the mask with the specified color
    rgb_fi[mask > 0.75] = np.asarray(
        [
            color[0] * mask[mask > 0.75],
            color[1] * mask[mask > 0.75],
            color[2] * mask[mask > 0.75],
        ]
    ).T

    # Return the new image
    return rgb_fi


# This probably needs to be merged with the above function!
# make sure to use deal with torch vs numpy arrays
def paint_image_tensor(
    image: torch.Tensor, masks: torch.Tensor, color: List[float]
) -> torch.Tensor:
    """
    Given an input image, a binary mask indicating where to paint, and a color to use,
    returns a new image where the pixels within the mask are colored with the specified color.

    Parameters
    ----------
    image: torch.Tensor
        Input image to paint.
    mask: torch.Tensor
        Binary mask indicating where to paint.
    color: List[float]
        List of 3 floats representing the RGB color to use.

    Returns
    -------
    rgb_body: torch.Tensor
        New image with painted pixels.
    """

    # Make a copy of the input image
    rgb_body = image.clone()

    c = 0
    for mask in masks:
        # Color the pixels within the mask with the specified color
        rgb_body[mask > 0.75] = torch.Tensor(
            [
                color[c][0] * mask[mask > 0.75],
                color[c][1] * mask[mask > 0.75],
                color[c][2] * mask[mask > 0.75],
            ]
        ).permute((1, 0))
        c += 1
    # Return the new image
    return rgb_body


# %%


def get_intersections(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Parameters
    ----------
    skeleton: np.ndarray
        Binary image of the skeleton

    Returns
    -------
        intersections: list
            List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validIntersection = [
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 1, 0, 0],
    ]

    image = skeleton.copy()
    intersections = []

    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1:
                nei = neighbours(x, y, image)
                valid = True
                if nei in validIntersection:
                    intersections.append((y, x))

    # DO IT OUTSIDE AS INDEPENDENT STEP
    # # Filter intersections to make sure we don't count them twice or ones that are very close together
    # for point1 in intersections:
    #     for point2 in intersections:
    #         if (
    #             ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10**2
    #         ) and (point1 != point2):
    #             intersections.remove(point2)
    # Remove duplicates
    intersections = list(set(intersections))
    return intersections


# %%
def get_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Given a skeletonised image, it will give the coordinates of the endpoints of the skeleton.

    Parameters
    ----------
    skeleton: numpy.ndarray
        The skeletonised image to detect the endpoints of

    Returns
    -------
    endpoints: list
        List of 2-tuples (x,y) containing the intersection coordinates
    """

    # A biiiiiig list of valid endpoints                 2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6
    validEndpoints = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]

    image = skeleton.copy()
    endpoints = []

    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1:
                nei = neighbours(x, y, image)
                if nei in validEndpoints:
                    endpoints.append((y, x))

    # Remove duplicates if any
    endpoints = list(set(endpoints))
    return endpoints


# %%
def neighbours(x, y, image):
    """
    Return 8-neighbours of image point P1(x,y), in a clockwise order

    Parameters
    ----------
    x: int
        x-coordinate of the point
    y: int
        y-coordinate of the point
    image: numpy.ndarray
        The image to find the neighbours of

    Returns
    -------
    _: list
        List of 8-neighbours of the point in the image
    """
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [
        img[x_1][y],
        img[x_1][y1],
        img[x][y1],
        img[x1][y1],
        img[x1][y],
        img[x1][y_1],
        img[x][y_1],
        img[x_1][y_1],
    ]


# %%
def traverse_graph(
    graph: dict, init: int, end_nodes: List[int], debug: bool = False
) -> List[List[int]]:
    """
    Function to traverse a graph from a starting node to a list of end nodes, and return all possible paths as a list of lists.

    Parameters
    ----------
    graph: dict
        The graph to traverse
    init: int
        The starting node ID
    end_nodes: list
        List of end nodes
    debug: bool
        Whether to print debug information

    Returns
    -------
    all_paths: list
        List of lists containing all possible paths from init to end_nodes

    TODO:
    Maybe.
    * Make it work for graphs with multiple paths between nodes and ensure that a subset of paths can be visited multiple times
    * Make a test for it
    """
    visited_ends = []
    core_vis = []
    all_paths = []
    path = []

    e = init
    if debug:
        print(f"start {e}")
    path.append(e)
    visited_ends.append(e)

    while True:
        neighs = graph[e]
        ends = [a for a in neighs if a in end_nodes]
        ends = [a for a in ends if a not in visited_ends]
        if ends:
            path.append(ends[0])
            all_paths.append(path)
            if debug:
                print(f"appending path {all_paths}, restart e={init}")
            visited_ends.append(ends[0])
            core_vis = []
            e = init
            path = []
            path.append(e)
        else:
            nex = [a for a in neighs if (a not in core_vis) and (a not in end_nodes)]
            if debug:
                print(f"nex {nex}")
            if len(nex) > 1:
                if nex[0] in path:
                    e = nex[1]
                else:
                    e = nex[0]
            else:
                if nex:
                    e = nex[0]
                else:
                    path.append(e)
                    all_paths.append(path)
                    return all_paths

            # track to avoid going back
            core_vis.append(e)
            path.append(e)
            if debug:
                print(f"nex {nex}, e {e}, core_vis {core_vis}")
                print(f"ends visited {visited_ends}")
                print(f"path so far {path}")

        if len(set(end_nodes).difference(set(visited_ends))) == 0:
            return all_paths


# %%
def segment_skel(skeleton, inter, conn=1):
    """
    Custom function to segment a skeletonised image into individual branches. Each branch gets a unique ID.

    Parameters
    ----------

    skeleton: numpy.ndarray
        The skeletonised image to segment
    inter: list
        List of 2-tuples (x,y) containing the intersection coordinates, as returned by the function find_intersections
    conn: int
        Connectivity of the skeleton. 1 for 4-connectivity, 2 for 8-connectivity

    Returns
    -------

    skel_labels: numpy.ndarray
        The labelled skeleton image
    edge_attributes: dict
        Dictionary containing the attributes of each edge (branch) (for now, its size in pixels)
    skprops: dict
        Dictionary containing the skimage.regionprops of each branch
    """

    zero_image = np.zeros_like(skeleton.copy()).astype(float)  # np.zeros_like(ssub)
    zero_image[np.asarray(inter)[:, 1], np.asarray(inter)[:, 0]] = 1
    zero_image[np.asarray(inter)[:, 1] + 1, np.asarray(inter)[:, 0]] = 1
    zero_image[np.asarray(inter)[:, 1], np.asarray(inter)[:, 0] + 1] = 1
    zero_image[np.asarray(inter)[:, 1] - 1, np.asarray(inter)[:, 0]] = 1
    zero_image[np.asarray(inter)[:, 1], np.asarray(inter)[:, 0] - 1] = 1
    if conn == 2:
        zero_image[np.asarray(inter)[:, 1] + 1, np.asarray(inter)[:, 0] + 1] = 1
        zero_image[np.asarray(inter)[:, 1] + 1, np.asarray(inter)[:, 0] - 1] = 1
        zero_image[np.asarray(inter)[:, 1] - 1, np.asarray(inter)[:, 0] + 1] = 1
        zero_image[np.asarray(inter)[:, 1] - 1, np.asarray(inter)[:, 0] - 1] = 1

    zero_image = np.clip(skeleton - zero_image, a_min=0, a_max=1)
    skel_labels = label(zero_image, connectivity=2)
    skprop = regionprops(skel_labels)

    edge_attributes = {}
    for i, r in enumerate(skprop):
        edge_attributes[i + 1] = r.area

    return skel_labels, edge_attributes, skprop


# %%
class Denormalize(object):
    """
    Denormalize a tensor image with mean and standard deviation, for plotting purposes.
    """

    def __init__(self, mean, std):
        """
        Parameters
        ----------
        mean: list
            List of mean values for each channel
        std: list
            List of standard deviation values for each channel
        """

        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        """
        Loads the image and applies the transformation to it.

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor image of size (C, H, W) to be normalized.

        Returns
        -------
        x_n: torch.Tensor
            Normalized image.
        """
        channel_dim = np.where([(a == 3) or (a == 1) for a in tensor.shape])[0]

        if channel_dim == 2:
            x_n = tensor.mul_(self.std).add_(self.mean)
            return x_n

        elif channel_dim == 0:
            for t, m, s in zip(tensor, self.mean, self.std):
                x_n = t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return x_n
