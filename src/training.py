#!/usr/bin/env python
# coding: utf-8

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# This code base can used to train the visual inspection/ defect analysis
# Deep learning model with and without data augmentation techniques returns:
# Time taken for training
# Evaluation metrics for Trained model
# Time taken for prediction on test dataset
#
# Model can be saved in 2 formats
# 1 - pytorch (.h5 ot .pth)
"""System module."""
# pylint: disable=E1101,E1102,E0401,R0914,R0801
import os
import sys
import time
import argparse
import itertools
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS


def run_hyperparams(n_folds=3, data_aug_flag=False, class_weigh=None, batch_s=10, device="cpu"):
    """run_hyperparameter function takes inputs of number of
    cross validation datasets and flag for data augmentation usage
    returns:
    Time taken for each param combination
    Total time taken for Hyperparameter tuning"""
    # since we are considering data augmentation of 5 types it will automatically consider batch_s*5
    if data_aug_flag:
        batch_s = int(batch_s/2)
    # hyperparams considered for tuning DL arch
    options = {
        "epochs": [1, 2, 3],
        "lr": [0.1, 0.01],
        "target_accuracy": [0.95, 0.98]}

    # Replicating GridsearchCV functionality for params generation
    keys = options.keys()
    values = (options[key] for key in keys)
    p_combinations = []
    for combination in itertools.product(*values):
        if len(combination) > 0:
            p_combinations.append(combination)

    print("Total number of fits = ", len(p_combinations)*n_folds)
    print("Take Break!!!\nThis will take time!")

    # Dataset loading
    cv_folds = get_cv_train_test_loaders(
        root=data_folder,
        batch_size=batch_s,
        n_folds=n_folds,
    )
    model_h = CustomVGG()
    class_weigh = torch.tensor(class_weigh).type(torch.FloatTensor).to(device)
    criterion_h = nn.CrossEntropyLoss(weight=class_weigh)
    start_total_time = time.time()
    ctr = 0
    for i, (train_loader1, test_loader1) in enumerate(cv_folds):
        for combination in p_combinations:
            if len(combination) > 0:
                ctr += 1
                print("Current fit is at ", ctr)
                epoch, learning_r, target_accuracy = combination
                print(f"Fold {i+1}/{n_folds} , epochs=", epoch, " learning rate=", learning_r,
                      " target accuracy=", target_accuracy)
                opt = optim.Adam(model_h.parameters(), lr=learning_r)
                start_t = time.time()
                model_h = train(train_loader1, model_h, opt, criterion_h, epoch, device,
                                target_accuracy=target_accuracy, data_aug=data_aug_flag)
                percv_time = time.time()-start_t
                print('time taker per cv=', percv_time)
                evaluate(model_h, test_loader1, device)
    total_cv_time = time.time()-start_total_time

    print('total_cv_time=', total_cv_time)


if __name__ == "__main__":
    # ## Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--datapath',
                        type=str,
                        required=False,
                        default='dataset/',
                        help='dataset path which consists of train and test folders')
    parser.add_argument('-o',
                        '--outmodel',
                        type=str,
                        required=False,
                        default='pill_intel',
                        help='outfile name without extension to save the model')
    parser.add_argument('-a',
                        '--dataaug',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling data augmentation, default is 0')
    parser.add_argument('-hy',
                        '--hyperparams',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 for enabling hyperparameter tuning, default is 0')
    FLAGS = parser.parse_args()

    data_folder = FLAGS.datapath
    subset_name = FLAGS.outmodel
    data_aug = FLAGS.dataaug
    hyperparams = FLAGS.hyperparams

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
    except FileNotFoundError:
        print("Please check the Data Path ")
        sys.exit()

    # Pramters used for the training DL model and hyperparameter Tuning
    BATCH_SIZE = 10
    TARGET_TRAINING_ACCURACY = 1.0
    LR = 0.0001
    EPOCHS = 1
    class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import intel_extension_for_pytorch as ipex
    DEVICE = "cpu"
    HEATMAP_THRESH = 0.7
    N_CV_FOLDS = 3

    # if hyperparameter tuning enabled default training code will not execute
    if hyperparams:
        print("Enabled Hyperparameter Tuning mode...for parameter changes refer Documentation!!")
        run_hyperparams(N_CV_FOLDS, data_aug, class_weight, BATCH_SIZE, DEVICE)
        sys.exit()

    # # Data
    train_loader, test_loader = get_train_test_loaders(
        root=data_folder, batch_size=BATCH_SIZE, test_size=0.2, random_state=42,
    )

    # Model Training
    # Intitalization of DL architechture along with optimizer and loss function
    model = CustomVGG()
    class_weight = torch.tensor(class_weight).type(
        torch.FloatTensor).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training module
    start_time = time.time()
    model = train(
        train_loader, model, optimizer, criterion, EPOCHS,
        DEVICE, TARGET_TRAINING_ACCURACY, data_aug=data_aug)
    train_time = time.time()-start_time

    # Model Saving
    model_path = f"{subset_name}"
    torch.save(model, model_path)

    print('train_time=', train_time)

    # # Visualization
    predict_localize(
        model, test_loader, DEVICE, thres=HEATMAP_THRESH, n_samples=15, show_heatmap=False
    )
