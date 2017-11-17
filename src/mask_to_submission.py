import glob
import os
import sys

import matplotlib.image as mpimg
import numpy as np

from constants import FOREGROUND_THRESHOLD


# assign a label to a patch
def patch_to_label(patch):
    return 1 if np.mean(patch) > FOREGROUND_THRESHOLD else 0


def mask_to_submission_strings(mask, idx):
    patch_size = 16
    for j in range(0, mask.shape[1], patch_size):
        for i in range(0, mask.shape[0], patch_size):
            patch = mask[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(idx, j, i, label))


def masks_to_submission(submission_filename, *masks):
    """Converts images into a submission file"""
    if os.path.exists(submission_filename):
        print("Delete old {} file".format(submission_filename))
        os.remove(submission_filename)

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for idx, mask in enumerate(masks):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(mask, idx))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: mask_to_submission.py csv_file_name mask_pattern')

    submission_filename = 'dummy_submission.csv'
    pattern = sys.argv[1]
    files = glob.glob(pattern)
    masks = [mpimg.imread(file) for file in files]
    masks_to_submission(sys.argv[0], *masks)
