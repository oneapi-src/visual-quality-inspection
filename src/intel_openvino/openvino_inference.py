"""System module."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# >> Based on the sample classification code from OpenVINO SDK
# pylint: disable=E1101,E1102,E0401,R0914,R0915
import sys
import os
import time
import argparse
import logging as log
import cv2
import numpy as np
from openvino.runtime import Core, PartialShape


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')

    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an '
                                                                     'image file(s).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, '
                           'MYRIAD, HDDL or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin '
                           'for device specified. Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. '
                                                               'Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. '
                                                                        'Number of top results.')
    return parser.parse_args()


def main():
    """
    main function for openvino
    """
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()

    # Step 1. Initialize runtime core
    log.info('Creating Core')
    core = Core()

    # Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format
    log.info("Reading the model: %s", args.model)
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model=args.model)  

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    # Step 3. Configure input & output
    log.info('Get input and output of model')
    input_layer = model.input(0)
    n_batch, channels, height, width = input_layer.shape

    # Get a number of input images   
    apath = os.path.abspath(args.input)
    files = os.listdir(apath)
    num_of_input = len(files)

    # Change batch size
    n_batch = num_of_input
    model.reshape([n_batch, channels, height, width])

    # Step 4. Compile model on the device
    log.info('Loading the model to the device')
    compiled_model = core.compile_model(model, args.device)

    # Step 5. Prepare input   
    input_data = np.ndarray(shape=(n_batch, channels, height, width))
    for i in range(num_of_input):
        image = cv2.imread(apath + '/' + files[i])

        if image.shape[:-1] != (height, width):
            image = cv2.resize(image, (width, height))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))

        input_data[i] = image

    # Step 6. Process output
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r', encoding="utf8") as file_labels:
            print([line.split(',')[0].strip() for line in file_labels])

    starttime = time.time()
    res = compiled_model([input_data])[compiled_model.output(0)]
    itime = time.time() - starttime

    # Processing output
    log.info("Processing output")
    # res = res[output_layer.any_name]
    print("Top ", args.number_top, " results: ")
    if args.labels:
        with open(args.labels, 'r', encoding="utf8") as file_labels:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in file_labels]
    else:
        labels_map = None

    classid_str = "classid"
    probability_str = "probability"

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        fname = apath + '/' + files[i]
        print("Image {}", fname)
        print(classid_str, probability_str)
        print('-' * len(classid_str), '-' * len(probability_str))
        for idx in top_ind:
            det_label = labels_map[idx] if labels_map else str(idx)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[idx]))) // 2
            print(' ' * space_num_before, det_label,
                  ' ' * space_num_after, ' ' * space_num_before_prob, probs[idx])

    log.info("Inference time for the batch of %s images %s secs.",
             str(num_of_input), str(itime))

    return 0


if __name__ == '__main__':
    sys.exit(main())
