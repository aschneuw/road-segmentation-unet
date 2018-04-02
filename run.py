#!/usr/bin/env python3

"""usage: ./run.py [GPU_ID]"""

import sys
if len(sys.argv) > 1:
    GPU_ID = int(sys.argv[1])
    print("Run on GPU {}".format(GPU_ID))
else:
    GPU_ID = -1
    print("Run on CPU")

#MODEL URL
MODEL_URL = "https://drive.switch.ch/index.php/s/TMNxxLWYfk61Jc5/download"

#MODEL SHA256
MODEL_SHA = "b0cf389d88b38494404693694e35dd4a2c316efad8cf948f59ad4e8528e00788"

# set correct environment variables
import os
os.environ[' CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
if GPU_ID == -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    os.environ['CUDA_VISIBLE_DEVICES'] ="{}".format(GPU_ID)

# add source to the Python path
import glob
import time
import urllib.request
import zipfile
import subprocess
import tensorflow as tf
import numpy as np

module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import images
from tf_aerial_images import Options
from tf_aerial_images import ConvolutionalModel
from constants import IMG_PATCH_SIZE, FOREGROUND_THRESHOLD


def get_model(path):
    """ makes sure model is on disk """
    path = os.path.abspath(os.path.join(path))

    if not os.path.exists(path):
        os.makedirs(path)

    modelpath = os.path.abspath(os.path.join(path, 'model.zip'))
    if not os.path.exists(modelpath):
        print("Download model:")
        print("===============")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(MODEL_URL, modelpath, report_download_progress)
        print()

    if len(glob.glob(path + "model-epoch-*")) < 3:
        print("Unzip model...")
        zip_ = zipfile.ZipFile(modelpath, 'r')
        zip_.extractall(path)
        zip_.close()


def verify_model():
    '''
    checks wheter the SHA for the downloaded model is valid or not
    :return: model SHA validity
    '''

    sha_file = 'model/model_sha.txt'
    try:
        return_code = subprocess.call("sha256sum model/model.zip > model/model_sha.txt", shell=True)

        if return_code != 0:
            #SHA256 Command failed
            return False

        if os.path.isfile('model/model_sha.txt'):
            # SHA FILE GENERATED
            with open(sha_file, 'r') as f:
                lines = f.readlines()

                if len(lines) > 0:
                    sha = lines[0].split(" ")[0]
                print("Computed SHA: {}".format(sha))

                if sha != MODEL_SHA:
                    print("SHA Verification for Model failed")
                    return False
                else:
                    print("SHA Verficiation for Model successful")
                    return True
        else:
            #SHA FILE not existent
            return False

    except:
        print("Unexpected error during SHA verification! Please verify manually")
        return False

def report_download_progress(count, block_size, total_size):
    """ callback to display download progress """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = np.maximum(time.time() - start_time, 1)
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...{}%, {:.1f} MB, {} KB/s, {:.0f} seconds passed".format(
        percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

# Set options
opts = Options()
opts.num_epoch = 0
opts.batch_size = 1
opts.pred_batch_size = 1
opts.patch_size = 388
opts.gpu = GPU_ID
opts.stride = 110
opts.num_layers = 6
opts.restore_model = True
opts.ensemble_prediction = True
opts.dilated_layers = True

# Path to final trained model
opts.model_path = "./model/model-epoch-011.chkpt"
opts.eval_data_dir = "./data/test/"
opts.save_path = "./prediction/"

# Make sure model is on disk
get_model(os.path.join(opts.save_path, "../model/"))

#Verify Model
is_valid = verify_model()

if not is_valid:
    print("Model verification failed! Probably because sha256sum is not installed on your machine or the download failed"
          " However, we do not abort... Please verify it manually!")
else:
    print("Automatic Model verification successful.")

# Run Prediction
if opts.gpu == -1:
    config = tf.ConfigProto()
else:
    config = tf.ConfigProto(device_count={'GPU': opts.num_gpu}, allow_soft_placement=True)

with tf.Graph().as_default(), tf.Session(config=config) as session:
    device = '/device:CPU:0' if opts.gpu == -1 else '/device:GPU:{}'.format(opts.gpu)
    print("Running on device {}".format(device))
    with tf.device(device):
        model = ConvolutionalModel(opts, session)

    # Restore model
    model.restore(file=opts.model_path)

    print("Running inference on eval data {}".format(opts.eval_data_dir))
    eval_images = images.load(opts.eval_data_dir)
    start = time.time()
    masks = model.predict_batchwise(eval_images, opts.pred_batch_size)
    stop = time.time()
    print("Prediction time:{} mins".format((stop - start)/60))

    masks = images.quantize_mask(masks, patch_size=IMG_PATCH_SIZE, threshold=FOREGROUND_THRESHOLD)
    overlays = images.overlays(eval_images, masks, fade=0.4)
    save_dir = os.path.abspath(os.path.join(opts.save_path, model.experiment_name))
    images.save_all(overlays, save_dir)
    images.save_submission_csv(masks, save_dir, IMG_PATCH_SIZE)