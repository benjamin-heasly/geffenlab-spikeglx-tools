# This runs in an Ubuntu environment.
FROM ubuntu:22.04

# Install wget and https certificate info, which we'll use to install things, below.
# Install git for obtaining SpikeGLX tools themselves.
# Clean up cache and temp files along the way.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install --no-install-recommends --yes wget ca-certificates git \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# Install conda.
RUN mkdir -p /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \
    && bash /opt/conda/miniconda.sh -b -p /opt/miniconda

# Install a specific version of TPrime.
ENV TPRIME_VERSION=Release_1_8
RUN git -C /opt clone --depth 1 --branch $TPRIME_VERSION https://github.com/billkarsh/TPrime.git
RUN cd /opt/TPrime/TPrime-linux && chmod +x install.sh && ./install.sh

# Install a specific version of CatGT.
ENV CATGT_VERSION=Release_4_7
RUN git -C /opt clone --depth 1 --branch $CATGT_VERSION https://github.com/billkarsh/CatGT.git
RUN cd /opt/CatGT/CatGT-linux && chmod +x install.sh && ./install.sh

# Install a specific version of C_Waves.
ENV CWAVES_VERSION=Release_2_9
RUN git -C /opt clone --depth 1 --branch $CWAVES_VERSION https://github.com/billkarsh/C_Waves.git
RUN cd /opt/C_Waves/C_Waves-linux && chmod +x install.sh && ./install.sh

# Create our conda environment from environment.yml.
COPY environment/environment.yml /opt/environment/environment.yml
RUN /opt/miniconda/bin/conda env create -f /opt/environment/environment.yml

# Include all of our code/* in /opt/code
COPY code /opt/code
COPY code/conda_run /usr/local/bin/conda_run
