# %% 
%load_ext autoreload
%autoreload 2

import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage import exposure
from sklearn.metrics import classification_report
from torchmetrics import ROC, ConfusionMatrix, F1Score, PrecisionRecallCurve

try:
    __IPYTHON__
except:
    prefix = ""  # or "../"
else:
    prefix = "../"  # or "../"

sys.path.append(f"{prefix}src")

from MZBLoader import Denormalize
from MZBModel import MZBModel
# from src.utils import read_pretrained_model, find_checkpoints
from utils import find_checkpoints, read_pretrained_model

# %% 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="rws33fgd", log="last")#.glob("**/*.ckpt")) 
# dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="z9gw5d0r", log="last")#.glob("**/*.ckpt")) 
dirs = find_checkpoints(Path(f"{prefix}mzb-pil"), version="rs66h9do", log="last")#.glob("**/*.ckpt")) 

mod_path = dirs[0]

model = MZBModel()  
model = model.load_from_checkpoint( 
        checkpoint_path=mod_path,
        # hparams_file= str(mod_path.parents[1] / 'hparams.yaml') #same params as args
    )

model.data_dir = Path(f"{prefix}data/learning_sets/")

model.eval()
# %%
# dataloader = model.train_dataloader(shuffle=False)
dataloader = model.dubendorf_dataloader(data_dir=Path("../data/clips_dubendorf/"))

pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    max_epochs=1,
    gpus=1, #[0,1],
    callbacks=[pbar_cb], 
    enable_checkpointing=False,
    logger=False
)

outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)

# %% 
snames, countnames = np.unique([a.name.split("_")[1] for a in dataloader.dataset.img_paths], return_counts=True)

dir_dict_dub = {}
for i, a in enumerate(snames):
    dir_dict_dub[a] = countnames[i]

for i, k in enumerate(dir_dict_dub):
    print(f"class {i}: {k} -- N = {dir_dict_dub[k]}")


dir_dict_trn = {}
ff = [a for a in sorted(list((model.data_dir / "trn_set").glob("*")))]
for i, a in enumerate(ff):
    dir_dict_trn[a.name] = a 

for i, k in enumerate(dir_dict_trn):
    print(f"class {i}: {k} -- N = {dir_dict_trn[k]}")


# %% 
y = []; p = []; gt = []
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


mzb_class = 4

denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

unc_score = -pc[:,mzb_class]
# unc_score = np.sum(pc * np.log(pc), axis=1)
# unc_score = np.sort(pc,axis=1)[:,-1] - np.sort(pc,axis=1)[:,-2]

# ss = np.argsort(-pc[:, mzb_class])
ss = np.argsort(unc_score)
sub = ss # [gc[ss] == mzb_class]

files = dataloader.dataset.img_paths

plt.figure(figsize=(15,5))
plt.plot(gc == mzb_class)
plt.plot(-unc_score)

# %% 
print(f"PREDICTING CLASS {list(dir_dict_trn.keys())[mzb_class]}") 

DETECT = 0
DIFF = True
FR = 0; N = 25

preds = []

for i, ti in enumerate(sub):
    if i < FR:
        continue

    fi = files[ti]
    im = Image.open(fi).convert("RGB")
    x = model.transform_ts(im)
    x = x[np.newaxis,...]

    with torch.set_grad_enabled(False):
        p = torch.softmax(model(x), dim = 1).cpu().numpy()
        pl_im = denorm(np.transpose(x.squeeze(),(1,2,0)))

    f, a = plt.subplots(1,1,figsize=(4,4))
    p_class = np.argmax(pc[ti,:])

    a.imshow(pl_im)
    a.set_title(f"Inference: predicted class {p_class} with P {pc[ti, p_class]:.2f}\n"\
    f"{files[ti]} \n GT {gc[ti]},  "\
    f"Y @ {pc[ti,mzb_class]:.2f}")
    
    # if pc[ti, mzb_class] < 0.01: 
        # break

    if i > N:
        break
# %% 
import shutil

err_folder = Path("../data/raw_learning_sets_duben/errors/")
err_folder.mkdir(exist_ok=True)
insect_folder = Path("../data/raw_learning_sets_duben/insects/")
insect_folder.mkdir(exist_ok=True)

print(f"MOVING INTO SUBFOLDER CLASS {list(dir_dict_trn.keys())[mzb_class]}") 
err_class = 4

for f, file in enumerate(files[:]): 

    print(f"{file}, {pc[f, err_class]:.2f}")

    if (pc[f, err_class] > 0.4) | (yc[f] == err_class):
        # print("error")
        shutil.copy(str(file), str(err_folder))
    else:
        shutil.copy(str(file), str(insect_folder))
        # print("insect")

# %% 
# Further refine "insects" in copy pasting them also in a folder derived from filename

dataloader = model.dubendorf_dataloader(data_dir=Path("../data/raw_learning_sets_duben/insects"))

pbar_cb = pl.callbacks.progress.TQDMProgressBar(refresh_rate=5)

trainer = pl.Trainer(
    max_epochs=1,
    gpus=1, #[0,1],
    callbacks=[pbar_cb], 
    enable_checkpointing=False,
    logger=False
)

outs = trainer.predict(model=model, dataloaders=[dataloader], return_predictions=True)

y = []; p = []; gt = []
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

# %% 

mzb_class = 4

denorm = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

unc_score = pc[:,mzb_class]
# unc_score = np.sum(pc * np.log(pc), axis=1)
# unc_score = np.sort(pc,axis=1)[:,-1] - np.sort(pc,axis=1)[:,-2]

# ss = np.argsort(-pc[:, mzb_class])
ss = np.argsort(unc_score)
sub = ss # [gc[ss] == mzb_class]

files = dataloader.dataset.img_paths

plt.figure(figsize=(15,5))
plt.plot(gc == mzb_class)
plt.plot(-unc_score)

# %% 
print(f"PREDICTING CLASS {list(dir_dict_trn.keys())[mzb_class]}") 

DETECT = 0
DIFF = True
FR = 0; N = 25

preds = []

for i, ti in enumerate(sub):
    if i < FR:
        continue

    fi = files[ti]
    im = Image.open(fi).convert("RGB")
    x = model.transform_ts(im)
    x = x[np.newaxis,...]

    with torch.set_grad_enabled(False):
        p = torch.softmax(model(x), dim = 1).cpu().numpy()
        pl_im = denorm(np.transpose(x.squeeze(),(1,2,0)))

    f, a = plt.subplots(1,1,figsize=(4,4))
    p_class = np.argmax(pc[ti,:])

    a.imshow(pl_im)
    a.set_title(f"Inference: predicted class {p_class} with P {pc[ti, p_class]:.2f}\n"\
    f"{files[ti]} \n GT {gc[ti]},  "\
    f"Y @ {pc[ti,mzb_class]:.2f}")
    
    # if pc[ti, mzb_class] < 0.01: 
        # break

    if i > N:
        break
# %%
mzb_taxonomy = pd.read_csv("../data/MZB_taxonomy.csv").drop(columns=["Unnamed: 0"]).set_index("query").ffill(axis=1)
reclass_into = "order"
remap_classes = [a.name.split("_")[1] for a in files]

print(mzb_taxonomy)

newnames = []
transform_names = {}

for i, na in enumerate(dir_dict_trn):
    transform_names[na] = i
transform_names["notthere"] = -1

# %% 
for name in remap_classes[:]:
    try:
        new_name = mzb_taxonomy.loc[name, reclass_into].lower()
    except: 
        new_name = "notthere"

    print(f"{name} into {new_name}")
    newnames.append(new_name)

# %%
gc_n = []
for nn in newnames: 
    gc_n.append(transform_names[nn])
gc_n = np.asarray(gc_n)

# %%
yhat_valid_labels = yc[gc_n > -1] 
gt_valid_labels = gc_n[gc_n > -1]

# %%
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             plot_confusion_matrix)

names = list(dir_dict_trn.keys())

cmat = confusion_matrix(gt_valid_labels, yhat_valid_labels, normalize="true", labels=np.arange(len(names)))

plt.figure(figsize=(7,7))
plt.imshow(cmat)
plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names, rotation=0)
plt.colorbar()

f = plt.figure(figsize=(10,10))
aa = f.gca() 
IC = type('IdentityClassifier', (), {"predict": lambda i : i, "_estimator_type": "classifier"})
plot_confusion_matrix(IC, yhat_valid_labels, gt_valid_labels, values_format=".1f", ax=aa,
     normalize=None, xticks_rotation="vertical", display_labels=names)
# %%
