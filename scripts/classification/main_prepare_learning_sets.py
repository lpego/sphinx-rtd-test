# %%
# This script takes the curated dataset (raw_learning_set) and prepares a refined learning set according to the taxonomy file.
# Each class will be aggregated to a single class according to the taxonomy file and where the cut (eg order, family, genus) is specified in the config file.
# The output structure is a directory tree with training and validation sets, each containing a folder for each class.

# The input directory tree is specified and curated by the user.

##### Need to wrap in main() ???

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mzbsuite.utils import cfg_to_arguments

# %%


def main(args, cfg):
    """
    Main function to prepare the learning sets for the classification task.

    Parameters
    ----------

    args: argparse.Namespace
        Arguments parsed from the command line. Specifically:
        
            - input_dir: path to the directory containing the raw clips
            - output_dir: path to the directory where the learning sets will be saved
            - config_file: path to the config file with train / inference parameters
            - taxonomy_file: path to the taxonomy file indicating classes per level
            - verbose: boolean indicating whether to print progress to the console

    cfg: dict
        Dictionary containing the configuration parameters.

    Returns
    -------
    None. Saves the files for the different splits in the specified folders.
    """

    np.random.seed(cfg.glob_random_seed)

    # rood of raw clip data
    root_data = Path(args.input_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # target folders definition
    target_trn = outdir / "trn_set/"
    target_val = outdir / "val_set/"

    # check if trn_set and val_set subfolders exist. If so, then interrupt the script.
    # This is to make sure that no overwriting happens; prompt the user that they need to specify a different output directory.
    if target_trn.exists() or target_val.exists():
        raise ValueError(
            # print in red and back to normal
            f"\033[91m Output directory {outdir} already exists. Please specify a different output directory.\033[0m"
        )

    # make dictionary to recode: key is current classification, value is target reclassif.
    # forward fill to get last valid entry and subset to desired column
    mzb_taxonomy = pd.read_csv(Path(args.taxonomy_file))
    if "Unnamed: 0" in mzb_taxonomy.columns:
        mzb_taxonomy = mzb_taxonomy.drop(columns=["Unnamed: 0"])
    mzb_taxonomy = mzb_taxonomy.ffill(axis=1)
    recode_order = dict(
        zip(mzb_taxonomy["query"], mzb_taxonomy[cfg.lset_class_cut].str.lower())
    )

    if args.verbose:
        print(f"Cutting phyl tree at {cfg.lset_class_cut}")

    # Move files to target folders for all files in the curated learning set
    for s_fo in recode_order:
        target_folder = target_trn / recode_order[s_fo]
        target_folder.mkdir(exist_ok=True, parents=True)

        for file in list((root_data / s_fo).glob("*")):
            shutil.copy(file, target_folder)

    # move out the validation set
    # make a small val set, 10% or 1 file, what is possible...
    size = cfg.lset_val_size
    trn_folds = [a.name for a in sorted(list(target_trn.glob("*")))]

    for s_fo in trn_folds:
        target_folder = target_val / s_fo
        target_folder.mkdir(exist_ok=True, parents=True)

        list_class = list((target_trn / s_fo).glob("*"))
        n_val_sam = np.max((1, np.ceil(0.1 * len(list_class))))

        val_files = np.random.choice(list_class, int(n_val_sam))

        if args.verbose:
            print(len(val_files), n_val_sam, len(list_class))

        for file in val_files:
            try:
                shutil.move(str(file), target_folder)
            except:
                print(f"{str(file)} into {target_folder}")

    # make a separate folder for the mixed set (actual test set)
    if (root_data / "mixed").is_dir():
        target_tst = outdir / "mixed_set/"
        target_tst.mkdir(exist_ok=True, parents=True)

        for file in list((root_data / "mixed").glob("*")):
            shutil.copy(file, target_tst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--taxonomy_file", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(str(args.config_file), "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = cfg_to_arguments(cfg)

    sys.exit(main(args, cfg))
