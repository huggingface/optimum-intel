FROM python:3.8

ARG commit_sha
ARG clone_url

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

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install npm@9.8.1 -g && \
    npm install n -g && \
    n latest

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/doc-builder.git
RUN git clone $clone_url && cd optimum-intel && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir ./optimum-intel[neural-compressor,openvino,nncf,quality]
