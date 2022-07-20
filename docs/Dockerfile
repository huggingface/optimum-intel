FROM nikolaik/python-nodejs:python3.8-nodejs18

RUN apt -y update
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir hf-doc-builder

RUN git clone -b documentation https://github.com/huggingface/optimum-intel.git
RUN python3 -m pip install --no-cache-dir ./optimum-intel[quality]