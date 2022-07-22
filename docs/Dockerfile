FROM ubuntu:18.04

# Need cv2 to install Intel Neural Compressor. See https://github.com/intel/neural-compressor/issues/36
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    gnupg \
    ffmpeg \
    libsm6 \
    libxext6

# Need node14 to build doc HTML. Taken from https://askubuntu.com/a/1113339
RUN curl -sL https://deb.nodesource.com/setup_14.x  | bash -
RUN apt-get -y install nodejs
RUN npm install

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN git clone https://github.com/huggingface/optimum-intel.git
RUN python3 -m pip install --no-cache-dir ./optimum-intel[quality]