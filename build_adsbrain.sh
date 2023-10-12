#!/bin/bash

# Note cmake version 3.27.6 works; but cmake version 3.25 will not work.
apt-get update
apt-get install libssl-dev
apt-get install rapidjson-dev
apt-get install libboost-all-dev
apt-get install libarchive-dev
nv-hostengine -t
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get install -y datacenter-gpu-manager
apt-get install libre2-dev
apt-get install -y cl-base64
apt-get install -y libb64-dev

rm -rf checksum/
rm -rf cmake_build
rm -rf opt/
rm -rf python
rm -rf tritonserver/
rm -rf onnxruntime

# Build without container
python build.py \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-metrics \
  --enable-gpu-metrics \
  --enable-gpu \
  --endpoint=http \
  --endpoint=adsbrain \
  --repo-tag=common:r22.05 \
  --repo-tag=core:r22.05 \
  --repo-tag=backend:r22.05 \
  --repo-tag=thirdparty:r22.05_ab \
  --backend=adsbrain:r22.05_ab \
  --backend=python:r22.05 \
  --backend=onnxruntime:r22.05 \
  --repoagent=checksum:r22.05 \
  --build-type=Release \
  --build-parallel=80 \
  --no-container-build \
  --build-dir=`pwd` \
  --verbose

# Build with container
python build.py \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-metrics \
  --enable-gpu-metrics \
  --enable-gpu \
  --endpoint=http \
  --endpoint=adsbrain \
  --repo-tag=common:r22.05 \
  --repo-tag=core:r22.05 \
  --repo-tag=backend:r22.05 \
  --repo-tag=thirdparty:r22.05_ab \
  --backend=adsbrain:r22.05_ab \
  --backend=python:r22.05 \
  --backend=onnxruntime \
  --repoagent=checksum:r22.05 \
  --build-type=Release \
  --build-parallel=80 \
  --verbose
