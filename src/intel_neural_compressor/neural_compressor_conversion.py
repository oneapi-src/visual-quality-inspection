# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""System module."""
# pylint: disable=E1101,E1102,E0401
import os
import sys
import argparse
import torch
from neural_compressor.experimental import Quantization
from utils.dataloader import get_train_test_loaders


if __name__ == "__main__":
    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--datapath',
                        type=str,
                        required=False,
                        default='dataset/',
                        help='dataset path which consists of train and test folders')
    parser.add_argument('-m',
                        '--modelpath',
                        type=str,
                        required=False,
                        default='pill_intel_model.h5',
                        help='Model path trained with pytorch ".h5" file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./conf.yaml',
                        help='Yaml file for quantizing model, default is "./config.yaml"')
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./output',
                        help='default output quantized model will be save in ./output folder')
    FLAGS = parser.parse_args()
    model_path = FLAGS.modelpath
    data_folder = FLAGS.datapath
    config_path = FLAGS.config
    out_path = FLAGS.outpath

    # Handle Exceptions for the user entries
    try:
        if os.path.exists(data_folder):
            print("Dataset path Found!!")
            if os.path.exists(os.path.join(data_folder, "train")) \
                    and os.path.exists(os.path.join(data_folder, "test")):
                print("Train and Test Data folders Found!")
            else:
                print("Please maintain data folder structure!")
        else:
            raise FileNotFoundError
        if os.path.exists(model_path):
            print("Model path Found!!")
        else:
            raise FileNotFoundError
        if os.path.exists(config_path):
            print("Config path Found!!")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("Please check the Path ")
        sys.exit()

    BATCH_SIZE = 10

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_train_test_loaders(
        root=data_folder, batch_size=BATCH_SIZE, test_size=0.2, random_state=42,
    )

    model = torch.load(model_path, map_location=DEVICE)

    # launch code for Intel® Neural Compressor
    quantizer = Quantization(config_path)
    quantizer.model = model
    quantizer.calib_dataloader = test_loader
    q_model = quantizer.fit()
    q_model.save(out_path)
