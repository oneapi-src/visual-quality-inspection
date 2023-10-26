# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

ARG IMAGE_NAME=intel/oneapi-aikit
ARG IMAGE_TAG=devel-ubuntu22.04
FROM ${IMAGE_NAME}:${IMAGE_TAG}
RUN apt-get update \
    && apt-get install -y --fix-missing --no-install-recommends \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

WORKDIR /workspace

ENV PYTHONPATH=/workspace

# Set Conda PATHs
ARG CONDA_INSTALL_PATH=/opt/conda
ARG CONDA_PREFIX=/opt/conda/envs/visual_inspection_intel
ARG MINICONDA_VERSION="latest"
# Miniconda Installation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_INSTALL_PATH} && \
    rm miniconda.sh && \
    ln -s ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    ${CONDA_INSTALL_PATH}/bin/conda clean --all && \
    echo ". ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate visual_inspection_intel" >> ~/.bashrc
# PATH prefers env over default conda
ENV PATH="${CONDA_PREFIX}/bin:${CONDA_INSTALL_PATH}/bin:${PATH}"

# Conda tools
RUN conda config --set solver libmamba \
    && conda env create -f /workspace/env/intel_env.yml \
    && conda create -n notebook_env jupyterlab nb_conda_kernels -q -y

