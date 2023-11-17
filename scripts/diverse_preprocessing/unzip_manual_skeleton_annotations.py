
# %% 
%load_ext autoreload
%autoreload 2 

import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
    PLOTS = False
else:
    prefix = "../../"  # or "../"
    PLOTS = True


# %% 
root_annotations = Path(f"{prefix}data/2021_swiss_invertebrates/")

if not (root_annotations / "manual_measurements").is_dir():

    files_manual = Path(f"{prefix}data/2021_swiss_invertebrates/manual_measurements-20230206T080300Z-001.zip")

    with zipfile.ZipFile(files_manual, 'r') as zip_ref:
        zip_ref.extractall(root_annotations)

root_annotations = root_annotations / "manual_measurements"
# %%
