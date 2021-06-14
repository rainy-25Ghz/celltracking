import json
import os
import datetime
import errno

import numpy as np
import deepcell
from deepcell.utils.tracking_utils import load_trks
from deepcell.utils.tracking_utils import save_trks


def save_trk(filename, lineages, raw, tracked):
    """Saves raw, tracked, and lineage data into one trks_file.

    Args:
        filename (str): full path to the final trk files.
        lineages (dict): a list of dictionaries saved as a json.
        raw (np.array): raw images data.
        tracked (np.array): annotated image data.

    Raises:
        ValueError: filename does not end in ".trks".
    """
    # if not str(filename).lower().endswith('.trks'):
    #     raise ValueError('filename must end with `.trks`. Found %s' % filename)
    os.mkdir(str(filename))
    with open(str(filename)+"/lineages.json", 'w+') as lineages_file:
        json.dump(lineages, lineages_file, indent=4)
        lineages_file.flush()
        lineages_file.close()
    with open(str(filename)+"/raw.npy", 'wb+') as raw_file:
        np.save(raw_file, raw)
        raw_file.flush()
        raw_file.close()
    with open(str(filename)+"/tracked.npy", 'wb+') as tracked_file:
        np.save(tracked_file, tracked)
        tracked_file.flush()
        tracked_file.close()


def load_trk(filename):
    with open(str(filename)+"/lineages.json") as f:
        lineages = json.load(f)
    with open(str(filename)+"/raw.npy", "rb") as f:
        raw = np.load(f)
    with open(str(filename)+"/tracked.npy", "rb") as f:
        tracked = np.load(f)
    return {'lineages': lineages, 'X': raw, 'y': tracked}
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
max_frames = 0
max_y = 0
max_x = 0
# max_frames = 30
# max_y = 154
# max_x = 182
# max_frames:C max_y:154 max_x:182
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
    lineages = []
    X = []
    y = []
    trks = load_trks(trks_file)
    for i, (lineage, raw, tracked) in enumerate(zip(trks['lineages'], trks['X'], trks['y'])):
        movie_counter = k + i
        print("i:%d  file:%s " % (i, trks_file))
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
    save_trk(str(k), lineages, X, y)
# # k = 0
# # movie_counter = 0
# # for trks_file in trks_files:
# #     trks = load_trks(trks_file)
# #     print("file name%s  " % (trks_file))
# #     lineages = []
# #     X = []
# #     y = []
# #     for i, (lineage, raw, tracked) in enumerate(zip(trks['lineages'], trks['X'], trks['y'])):
# #         movie_counter = k + i
# #         print("i:%d" % (i))
# #         # Normalize the raw images
# #         for frame in range(raw.shape[0]):
# #             raw[frame, :, :, 0] = image_norm(raw[frame, :, :, 0])

# #         # Image padding if neccesary - This assumes that raw and tracked have the same shape
# #         if raw.shape[1] < max_y:
# #             diff2pad = max_y - raw.shape[1]
# #             pad_width = int(diff2pad / 2)
# #             if diff2pad % 2 == 0:
# #                 # Pad width can be split evenly
# #                 raw = np.pad(raw, ((0, 0), (pad_width, pad_width),
# #                              (0, 0), (0, 0)), mode='constant', constant_values=0)
# #                 tracked = np.pad(tracked, ((0, 0), (pad_width, pad_width),
# #                                  (0, 0), (0, 0)), mode='constant', constant_values=0)
# #             else:
# #                 # Pad width cannot be split evenly
# #                 raw = np.pad(raw, ((0, 0), (pad_width + 1, pad_width),
# #                              (0, 0), (0, 0)), mode='constant', constant_values=0)
# #                 tracked = np.pad(tracked, ((0, 0), (pad_width + 1, pad_width),
# #                                  (0, 0), (0, 0)), mode='constant', constant_values=0)

# #         if raw.shape[2] < max_x:
# #             diff2pad = max_x - raw.shape[2]
# #             pad_width = int(diff2pad / 2)
# #             if diff2pad % 2 == 0:
# #                 # Pad width can be split evenly
# #                 raw = np.pad(raw, ((0, 0), (0, 0), (pad_width, pad_width),
# #                              (0, 0)), mode='constant', constant_values=0)
# #                 tracked = np.pad(tracked, ((
# #                     0, 0), (0, 0), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
# #             else:
# #                 # Pad width cannot be split evenly
# #                 raw = np.pad(raw, ((0, 0), (0, 0), (pad_width+1, pad_width),
# #                              (0, 0)), mode='constant', constant_values=0)
# #                 tracked = np.pad(tracked, ((0, 0), (0, 0), (pad_width+1,
# #                                  pad_width), (0, 0)), mode='constant', constant_values=0)

# #         if raw.shape[0] < max_frames:
# #             pad_width = int(max_frames-raw.shape[0])
# #             raw = np.pad(raw, ((0, pad_width), (0, 0), (0, 0),
# #                          (0, 0)), mode='constant', constant_values=0)
# #             tracked = np.pad(tracked, ((0, pad_width), (0, 0),
# #                              (0, 0), (0, 0)), mode='constant', constant_values=0)

# #         lineages.append(lineage)
# #         X.append(raw)
# #         y.append(tracked)

# #     k = movie_counter + 1
# #     save_trk("%d" % (k), lineages=lineage, raw=X, tracked=y)
# list = ['420', '679', '803']
# trk = load_trk('240')
# lineages .append(trk["lineages"])
# X = trk["X"]
# y = trk["y"]
# for dir in list:
#     trk = load_trk(dir)
#     lineages.append(trk["lineages"])
#     X.concatenate(trk["X"])
#     y.concatenate(trk["y"])
# # for dir in list:
# #     trk = load_trk(dir)
# #     lineages.append(trk["lineages"])
# #     X.append(trk["raw"])
# #     y.append(trk["tracked"])
# # # Save the combined datasets into one trks file
# filename = 'combined_data.trks'
# save_trks(os.path.join(basepath, filename), lineages, X, y)
