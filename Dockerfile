FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Setup normal package
RUN apt update && \
    apt install -y g++ gcc make unzip git software-properties-common wget libpng-dev libjpeg-dev python3 python3-dev && \
    apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates

# Install node
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash - && \
    apt -y install nodejs

# Install libss
RUN echo "deb http://security.ubuntu.com/ubuntu bionic-security main" | tee -a /etc/apt/sources.list.d/bionic.list && \
    apt update && \
    apt-cache policy libssl1.0-dev && \
    apt-get install -y libssl1.0-dev

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y cmake

RUN npm set unsafe-perm true && npm install -g yarn

# Install torchjs
COPY . /root/nodejs/
WORKDIR /root/nodejs
RUN yarn upgrade --dev && \
    yarn build-prebuild && \
    yarn build && \
    yarn install && \
    yarn test

