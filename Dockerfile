# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.0-base
RUN apt update

# set it to run as non-interactive
ARG DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# update/upgrade apt
ENV TZ=Europe/Paris
RUN apt upgrade -y

#install basics
RUN apt-get install git-all -yq
RUN apt-get install curl -yq

#install miniconda
ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh Miniconda3.sh
RUN bash Miniconda3.sh -b -p /miniconda
ENV PATH="/miniconda/bin:${PATH}"
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN rm Miniconda3.sh
RUN conda install wget -y

# create and activate conda environment
RUN conda create -q -n run-environement python="3.7" numpy scipy matplotlib
RUN /bin/bash -c "source activate run-environement"
RUN conda install python="3.7" pip

# install basics
RUN apt-get install less nano -yq
RUN apt-get -yq install build-essential
RUN apt-get -yq install libsuitesparse-dev
RUN apt-get -yq install wget
RUN apt-get -yq install unzip
RUN apt-get -yq install lsof
RUN apt-get update
RUN apt-get -yq install libsm6 libxrender1 libfontconfig1 libglib2.0-0

# install torch and co
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# install mongo internally
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list
RUN apt-get update
RUN apt-get install -y mongodb-org
RUN mkdir /data
RUN mkdir /data/db
RUN mongod --fork --logpath /dev/null
ENV MONGOROOTPASS=''

# install pymongo
RUN pip install pymongo

# check that scipy, numpy and matplotlib are well installed
RUN conda install python="3.7" scipy numpy matplotlib

# check nvidia drivers are working
CMD nvidia-smi
