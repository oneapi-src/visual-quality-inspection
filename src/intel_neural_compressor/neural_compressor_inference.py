# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""System module."""
# pylint: disable=E1101,E1102,E0401
import os
import sys
import time
import argparse
import torch
from neural_compressor.utils.pytorch import load
from neural_compressor.experimental import Benchmark
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
    parser.add_argument('-fp32',
                        '--fp32modelpath',
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
    parser.add_argument('-int8',
                        '--int8modelpath',
                        type=str,
                        required=False,
                        default='./output',
                        help='load the quantized model folder. default is ./output folder')
    parser.add_argument('-b',
                        '--batchsize',
                        type=int,
                        required=False,
                        default=1,
                        help='use the batchsize that want do inference, default is 1')
    FLAGS = parser.parse_args()
    fp_model = FLAGS.fp32modelpath
    data_folder = FLAGS.datapath
    config_path = FLAGS.config
    int_model = FLAGS.int8modelpath
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
        if not os.path.exists(fp_model):
            print("FP32 Model path Not Found!!")
            raise FileNotFoundError
        if not os.path.exists(int_model):
            print("INT8 Model path Not Found!!")
            raise FileNotFoundError
        if not os.path.exists(config_path):
            print("Config path Not Found!!")
            raise FileNotFoundError
    except FileNotFoundError:
        print("Please check the Paths")
        sys.exit()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_train_test_loaders(
        root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42,
    )

    model = torch.load(fp_model, map_location=DEVICE)

    int8_model = load(int_model, model)
    int8_model.to(DEVICE)
    int8_model.eval()
    model.eval()

    # Timing Analysis
    FP_AVG_TIME = 0
    INT_AVG_TIME = 0
    COUNT = 0
    for images in next(iter(test_loader)):
        if len(images.shape) > 2:
            COUNT += 1
            start_time = time.time()
            model(images)
            pred_time_fp = time.time() - start_time
            start_time = time.time()
            int8_model(images)
            pred_time_int = time.time() - start_time
            FP_AVG_TIME += pred_time_fp
            INT_AVG_TIME += pred_time_int
    print("Batch Size used here is ", batch_size)
    print("Average Inference Time Taken Fp32 --> ", (FP_AVG_TIME / COUNT))
    print("Average Inference Time Taken Int8 --> ", (INT_AVG_TIME / COUNT))

    #  EVALUATION
    print("*"*50)
    print("Evaluating the Quantizaed Model")
    print("*"*50)

    evaluator = Benchmark(config_path)
    evaluator.model = int8_model
    # create benchmark dataloader like examples/tensorflow/qat/benchmark.py
    evaluator.b_dataloader = test_loader
    evaluator('accuracy')

    print("*"*50)
    print("Evaluating the FP32 Model")
    print("*"*50)
    evaluator = Benchmark(config_path)
    evaluator.model = model
    # create benchmark dataloader like examples/tensorflow/qat/benchmark.py
    evaluator.b_dataloader = test_loader
    evaluator('accuracy')
