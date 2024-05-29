<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Run language generation with deepspeed

This script run IPEXModel with deepspeed AutpTP.

Please run the fowllowing commands to setup environment.

Docker-based environment setup
```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.3.0+cpu
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile by installing from Intel® Extension for PyTorch\* prebuilt wheel files
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.0 .

# Run the container with command below, please mount the current folder or copy the python script `run_generation_deepspeed.py`
docker run --rm -it --privileged ipex-llm:2.3.0 bash

# When the command prompt shows inside the docker container, enter llm examples directory
cd llm

# Activate environment variables
source ./tools/env_activate.sh
```

Conda-based environment setup
```bash
# Get the Intel® Extension for PyTorch\* source code
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch
git checkout v2.3.0+cpu
git submodule sync
git submodule update --init --recursive

# GCC 12.3 is required. Installation can be taken care of by the environment configuration script.
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm

# Setup the environment with the provided script
# A sample "prompt.json" file for benchmarking is also downloaded
cd examples/cpu/inference/python/llm
bash ./tools/env_setup.sh 7

# Activate environment variables
source ./tools/env_activate.sh
```

Then run the script with the following command.
```bash
deepspeed --bind_cores_to_rank run_generation_deepspeed.py
```