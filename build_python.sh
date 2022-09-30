#!/bin/bash

# apt-get update
# apt-get install libssl-dev
# apt-get install rapidjson-dev
# nv-hostengine -t
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
# wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
# dpkg -i cuda-keyring_1.0-1_all.deb
# apt-get install -y datacenter-gpu-manager
# apt install libre2-dev
# apt-get install -y cl-base64
# apt-get install -y libb64-dev

python build.py \
  --no-container-build \
  --build-dir=/datadrive/fhu/github/triton-server-abo-v2/triton-server/ \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-metrics \
  --enable-gpu-metrics \
  --enable-gpu \
  --endpoint=http \
  --endpoint=adsbrain \
  --repo-tag=common:main \
  --repo-tag=core:main \
  --repo-tag=backend:main \
  --repo-tag=thirdparty:main \
  --backend=python:main \
  --repoagent=checksum:main \
  --build-type=Debug \
  --build-parallel=80 \
  --verbose
