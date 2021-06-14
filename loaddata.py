import json
import os
import datetime
import errno

import numpy as np
import deepcell
from deepcell.utils.tracking_utils import load_trks
from deepcell.utils.tracking_utils import save_trks


def load_trk(filename):
    with open(str(filename)+"/lineages.json") as f:
        lineages = json.load(f)
    with open(str(filename)+"/raw.npy", "rb") as f:
        raw = np.load(f)
    with open(str(filename)+"/tracked.npy", "rb") as f:
        tracked = np.load(f)
    return {'lineages': lineages, 'X': raw, 'y': tracked}


list = ['240', '420', '679', '803']
lineages = []
for i, folder in enumerate(list):
    t = load_trk(folder)
    if i == 0:
        # lineages = t["lineages"]
        X = t["X"]
        y = t["y"]
    else:
        # lineages = lineages+t["lineages"]
        X = np.concatenate((X, t["X"]), axis=0)
        y = np.concatenate((y, t["y"]), axis=0)
# save_trks()
filename = 'combined_data.trks'
basepath = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
save_trks(os.path.join(basepath, filename), lineages, X, y)
