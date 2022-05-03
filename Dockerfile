FROM ubuntu:focal-20220415 as build

RUN apt update && \
    apt install -y python3-dev \
        python3-pip \
        wget

RUN pip install wheel

RUN wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

WORKDIR /build

COPY setup.py .
COPY torchsr torchsr
COPY README.md .

RUN python3 setup.py bdist_wheel

FROM nvcr.io/nvidia/cuda:11.3.0-base-ubuntu20.04

WORKDIR /torchsr

RUN apt update && \
    apt install -y python3-pip

COPY --from=build vgg19-dcbb9e9d.pth /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
COPY --from=build /build/dist .
COPY requirements.txt .
COPY media/waterfalls-low-res.png media/waterfalls-low-res.png

RUN pip3 install --no-cache-dir \
    -r requirements.txt \
    python-hostlist \
    torchsr-0.1.0-py3-none-any.whl && \
    rm torchsr-0.1.0-py3-none-any.whl

RUN mkdir -p output && \
    rm requirements.txt
