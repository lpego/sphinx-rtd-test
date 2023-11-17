import torch
import numpy as np

# from torchvision import utils
from PIL import Image  # , ImageFilter, ImageDraw, ImageOps

from torch.utils.data import Dataset

from mzbsuite.skeletons.mzb_skeletons_helpers import Denormalize


class MZBLoader_skels(Dataset):
    """
    Class definition for the dataloader for the macrozoobenthos skeletons dataset.

    Parameters
    ----------
    im_folder : Path
        folder path of input images
    bo_folder : Path
        folder path of body length skeleton masks
    he_folder : Path
        folder path of head length skeleton masks
    ls_inds : list
        indices of images to be used for the learning set, optional
    learning_set : str
        type of learning set to be used, optional, default: 'all'
    transforms : torchvision.transforms
        optional, default: None
    """

    def __init__(
        self,
        im_folder,
        bo_folder,
        he_folder,
        ls_inds=[],
        learning_set="all",
        transforms=None,
    ):
        self.transforms = transforms
        self.imsize = 224

        self.ls_inds = ls_inds
        self.learning_set = learning_set
        self.im_folder = im_folder
        self.bo_folder = bo_folder
        self.he_folder = he_folder

        self.denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.img_paths, self.mbo_paths, self.mhe_paths, self.inds = self.prepare_data(
            self.im_folder, self.bo_folder, self.he_folder, ls_inds=self.ls_inds
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        f = self.img_paths[idx]
        img = Image.open(f).convert("RGB")
        # THIS IS A HACK
        seed = np.random.randint(123456789)  # make a seed with numpy generator

        # THIS IS A HACK
        np.random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transforms is not None:
            tensor_image = self.transforms(img)

        if len(self.mbo_paths) > 0:
            f = self.mbo_paths[idx]
            mbo = Image.open(f).convert("RGB")
            # THIS IS A HACK
            np.random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            if self.transforms is not None:
                tensor_bmsk = self.transforms(mbo)

            tensor_bmsk = self.denorm(tensor_bmsk)
            tensor_bmsk = torch.gt(tensor_bmsk, 0).long()
        else:
            tensor_bmsk = torch.zeros_like(tensor_image)

        if len(self.mhe_paths) > 0:
            f = self.mhe_paths[idx]
            mhe = Image.open(f).convert("RGB")
            # THIS IS A HACK
            np.random.seed(seed)  # apply this seed to target tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            if self.transforms is not None:
                tensor_hmsk = self.transforms(mhe)

            tensor_hmsk = self.denorm(tensor_hmsk)
            tensor_hmsk = torch.gt(tensor_hmsk, 0).long()
        else:
            tensor_hmsk = torch.zeros_like(tensor_image)

        if (len(self.mbo_paths) > 0) and (len(self.mhe_paths) > 0):
            # tensor_hmsk[tensor_hmsk != 0] += 1
            tensor_mask = torch.stack((tensor_bmsk, tensor_hmsk), dim=0)
        else:
            tensor_mask = torch.zeros_like(tensor_image[0, ...])
        # tensor_mask = torch.clamp(tensor_mask, min=0, max=2).long()
        # tensor_mask = torch.LongTensor(tensor_mask)  # [None, ...]

        return tensor_image, tensor_mask, idx

    @staticmethod
    def prepare_data(im_folder, bo_folder=None, he_folder=None, ls_inds=[]):
        """
        Prepares the data for the dataloader, loads it and returns it as numpy arrays.
        """

        # At least one mask folder needs to exist
        # assert bo_folder.is_dir() or he_folder.is_dir()

        # this makes a one folder - one class connection, and prepares data arrays consequently
        images = np.asarray(sorted(list(im_folder.glob("*.jpg"))))

        if bo_folder.is_dir():
            mbody = np.asarray(sorted(list(bo_folder.glob("*.jpg"))))
        else:
            mbody = []

        if he_folder.is_dir():
            mhead = np.asarray(sorted(list(he_folder.glob("*.jpg"))))
        else:
            mhead = []

        if len(ls_inds) > 0:
            return images[ls_inds], mbody[ls_inds], mhead[ls_inds], ls_inds
        else:
            return images, mbody, mhead, ls_inds
