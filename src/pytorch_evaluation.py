# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""System module."""
# pylint: disable=E1101,E1102,E0401
import os
import argparse
import sys
import time
import torch
from utils.dataloader import get_train_test_loaders
from utils.helper import evaluate


if __name__ == "__main__":
    # This code will evaluate the trained model on entire test dataset and produce the evaluation
    # metrics
    # returns:
    # Accuracy
    # Prediction Time

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_folder',
                        type=str,
                        required=False,
                        default='dataset',
                        help='dataset path which consists of train and test folders')
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=False,
                        default='weights/pill_intel_model.h5',
                        help='Absolute path to the h5 pytorch model with extension ".h5"')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=1,
                        help='use the batchsize that want do inference, default is 1')
    FLAGS = parser.parse_args()

    data_folder = FLAGS.data_folder
    model_path = FLAGS.model_path
    batch_size = FLAGS.batchsize

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
        if not os.path.exists(model_path):
            print("Model path Not Found!!")
            raise FileNotFoundError
    except FileNotFoundError:
        print("Please check the Paths Provided")
        sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import intel_extension_for_pytorch as ipex
    DEVICE = "cpu"
    # Load dataset
    start_time = time.time()
    train_loader, test_loader = get_train_test_loaders(
        root=data_folder, batch_size=batch_size, test_size=0.28, random_state=42,
    )
    load_time = time.time()-start_time
    # Loading Model
    model = torch.load(model_path, map_location="cpu")
    # Evaluation
    # evaluate(model, test_loader, device)
    # Inferencing
    start_time = time.time()
    evaluate(model, train_loader, device)
    pred_time = time.time()-start_time
    print('pred_time=', pred_time)
