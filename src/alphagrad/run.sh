#!/bin/zsh
docker run --rm -it --gpus all --cpus=8 --memory 16G --user $(id -u):$(id -g) --privileged \
--mount type=bind,source=/Users/grieser,destination=/home/user \
alphagrad:latest \
bash