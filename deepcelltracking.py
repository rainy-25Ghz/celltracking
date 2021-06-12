import os
import datetime
import errno

import numpy as np
import deepcell
from deepcell.utils.tracking_utils import load_trks
from deepcell.utils.tracking_utils import save_trks

# filename_HeLa = 'HeLa_S3.trks'
# (X_train, y_train), (X_test,
#                      y_test) = deepcell.datasets.tracked.hela_s3.load_tracked_data(filename_HeLa)
# print('HeLa -\nX.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))


filename_3T3 = '3T3_NIH.trks'
filename_HeLa = 'HeLa_S3.trks'
filename_HEK = 'HEK293.trks'
filename_RAW = 'RAW2647.trks'

# Define all the trks to load
basepath = os.path.expanduser(os.path.join('~', '.keras', 'datasets'))
trks_files = [os.path.join(basepath, filename_3T3),
              os.path.join(basepath, filename_HeLa),
              os.path.join(basepath, filename_HEK),
              os.path.join(basepath, filename_RAW)]

# Each TRKS file may have differrent dimensions,
# but the model expects uniform dimensions.
# Determine max dimensions and zero pad as neccesary.
max_frames = 1
max_y = 1
max_x = 1

for trks_file in trks_files:
    trks = load_trks(trks_file)

    # Store dimensions of raw and tracked
    # to check new data against to pad if neccesary
    if trks['X'][0].shape[0] > max_frames:
        max_frames = trks['X'][0].shape[0]
    if trks['X'][0].shape[1] > max_y:
        max_y = trks['X'][0].shape[1]
    if trks['X'][0].shape[2] > max_x:
        max_x = trks['X'][0].shape[2]

print("max_frames:%d max_y:%d max_x:%d" % (max_frames, max_y, max_x))
# Define a normalizaiton function for the raw images that can be run before padding


def image_norm(original_image):
    # NNs prefer input data that is 0 mean and unit variance
    normed_image = (original_image - np.mean(original_image)) / \
        np.std(original_image)
    return normed_image


# Load each trks file, normalize and pad as neccesary
lineages = []
X = []
y = []

k = 0
movie_counter = 0
for trks_file in trks_files:
    trks = load_trks(trks_file)
    for i, (lineage, raw, tracked) in enumerate(zip(trks['lineages'], trks['X'], trks['y'])):
        movie_counter = k + i

        # Normalize the raw images
        for frame in range(raw.shape[0]):
            raw[frame, :, :, 0] = image_norm(raw[frame, :, :, 0])

        # Image padding if neccesary - This assumes that raw and tracked have the same shape
        if raw.shape[1] < max_y:
            diff2pad = max_y - raw.shape[1]
            pad_width = int(diff2pad / 2)
            if diff2pad % 2 == 0:
                # Pad width can be split evenly
                raw = np.pad(raw, ((0, 0), (pad_width, pad_width),
                             (0, 0), (0, 0)), mode='constant', constant_values=0)
                tracked = np.pad(tracked, ((0, 0), (pad_width, pad_width),
                                 (0, 0), (0, 0)), mode='constant', constant_values=0)
            else:
                # Pad width cannot be split evenly
                raw = np.pad(raw, ((0, 0), (pad_width + 1, pad_width),
                             (0, 0), (0, 0)), mode='constant', constant_values=0)
                tracked = np.pad(tracked, ((0, 0), (pad_width + 1, pad_width),
                                 (0, 0), (0, 0)), mode='constant', constant_values=0)

        if raw.shape[2] < max_x:
            diff2pad = max_x - raw.shape[2]
            pad_width = int(diff2pad / 2)
            if diff2pad % 2 == 0:
                # Pad width can be split evenly
                raw = np.pad(raw, ((0, 0), (0, 0), (pad_width, pad_width),
                             (0, 0)), mode='constant', constant_values=0)
                tracked = np.pad(tracked, ((
                    0, 0), (0, 0), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
            else:
                # Pad width cannot be split evenly
                raw = np.pad(raw, ((0, 0), (0, 0), (pad_width+1, pad_width),
                             (0, 0)), mode='constant', constant_values=0)
                tracked = np.pad(tracked, ((0, 0), (0, 0), (pad_width+1,
                                 pad_width), (0, 0)), mode='constant', constant_values=0)

        if raw.shape[0] < max_frames:
            pad_width = int(max_frames-raw.shape[0])
            raw = np.pad(raw, ((0, pad_width), (0, 0), (0, 0),
                         (0, 0)), mode='constant', constant_values=0)
            tracked = np.pad(tracked, ((0, pad_width), (0, 0),
                             (0, 0), (0, 0)), mode='constant', constant_values=0)

        lineages.append(lineage)
        X.append(raw)
        y.append(tracked)

    k = movie_counter + 1

# Save the combined datasets into one trks file
filename = 'combined_data.trks'
save_trks(os.path.join(basepath, filename), lineages, X, y)
