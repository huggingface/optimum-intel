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

This script shows how to run IPEXModel with deepspeed AutpTP.

Please run the fowllowing commands to setup environment.

1. Get the ipex docker image: `docker pull intel/intel-extension-for-pytorch:2.3.0-pip-base`.
2. Go into the container and install oneapi and other libs by the following scirpt:
```bash
apt update && apt install -y gpg-agent

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

apt-get update && apt install -y intel-basekit numactl && apt-get install -y python3-dev git

pip install transformers deepspeed accelerate

source /opt/intel/oneapi/setvars.sh
```
3. Install optimum-intel.
4. Run the script with the following command.
```bash
deepspeed --bind_cores_to_rank run_generation_deepspeed.py
```