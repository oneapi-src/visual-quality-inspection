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
from openvino.inference_engine import IECore


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    args.add_argument('-i', '--input', required=True, type=str, help='Required. Path to an '
                                                                     'image file(s).')
    args.add_argument('-l', '--extension', type=str, default=None,
                      help='Optional. Required by the CPU Plugin for executing the custom '
                           'operation on a CPU. '
                      'Absolute path to a shared library with the kernels implementations.')
    args.add_argument('-c', '--config', type=str, default=None,
                      help='Optional. Required by GPU or VPU Plugins for the custom '
                           'operation kernel. '
                      'Absolute path to operation description file (.xml).')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, '
                           'MYRIAD, HDDL or HETERO: '
                           'is acceptable. The sample will look for a suitable plugin '
                           'for device specified. Default value is CPU.')
    args.add_argument('--labels', default=None, type=str, help='Optional. '
                                                               'Path to a labels mapping file.')
    args.add_argument('-nt', '--number_top', default=10, type=int, help='Optional. '
                                                                        'Number of top results.')
    # fmt: on
    args.add_argument('--outputname', default='64', type=str, help='Optional. '
                                                                   'Output blob name '
                                                                   'for the classification.')
    return parser.parse_args()


def main():
    """
    main function for openvino
    """
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = parse_args()
    out_blob = args.outputname

    # Step 1. Initialize inference engine core
    log.info('Creating Inference Engine')
    inference_engine = IECore()

    if args.extension and args.device == 'CPU':
        log.info("Loading the %s extension: %s", args.device, args.extension)
        inference_engine.add_extension(args.extension, args.device)

    if args.config and args.device in ('GPU', 'MYRIAD', 'HDDL'):
        log.info("Loading the %s configuration: %s", args.device, args.config)
        inference_engine.set_config({'CONFIG_FILE': args.config}, args.device)

    # Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format
    log.info("Reading the network: %s", args.model)
    # (.xml and .bin files) or (.onnx file)
    net = inference_engine.read_network(model=args.model)

    if len(net.input_info) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    # Step 3. Configure input & output
    log.info('Configuring input and output blobs')
    # Get names of input and output blobs
    input_blob = next(iter(net.input_info))

    # Set input and output precision manually
    net.input_info[input_blob].precision = 'FP32'
    net.outputs[out_blob].precision = 'FP32'

    # Get a number of input images
    apath = os.path.abspath(args.input)
    files = os.listdir(apath)
    num_of_input = len(files)
    net.batch_size = num_of_input
    # num_of_input = len(args.input)
    # Get a number of classes recognized by a model
    # num_of_classes = max(net.outputs[out_blob].shape)

    # Step 4. Loading model to the device
    log.info('Loading the model to the plugin')
    exec_net = inference_engine.load_network(network=net, device_name=args.device)

    # Step 5. Create infer request
    # load_network() method of the IECore class with a
    # specified number of requests (default 1) returns an ExecutableNetwork
    # instance which stores infer requests. So you already created
    # Infer requests in the previous step.

    # Step 6. Prepare input
    # input_data = []
    n_batch, channels, height, width = net.input_info[input_blob].input_data.shape
    input_data = np.ndarray(shape=(n_batch, channels, height, width))

    for i in range(num_of_input):
        image = cv2.imread(apath + '/' + files[i])

        if image.shape[:-1] != (height, width):
            image = cv2.resize(image, (width, height))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        # Add N dimension to transform to NCHW
        image = np.expand_dims(image, axis=0).astype(np.float32)

        input_data[i] = image

    # Step 7. Process output
    # Generate a label list
    if args.labels:
        with open(args.labels, 'r', encoding="utf8") as file_labels:
            print([line.split(',')[0].strip() for line in file_labels])

    starttime = time.time()
    res = exec_net.infer(inputs={input_blob: input_data})
    itime = time.time() - starttime

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
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
