# FROM ghcr.io/feelpp/feelpp:v0.110.2-focal
# feelpp:v0.111.0-alpha.5-bullseye
# not working
FROM ubuntu:noble


USER root

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    OMPI_MCA_btl_vader_single_copy_mechanism=none

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.
ARG USERNAME=feelpp
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# for singularity
ARG SVERSION=3.7.4
ARG GVERSION=1.14.12
ARG OS=linux
ARG ARCH=amd64

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Setup demo environment variables
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8


# install Feelpp from Feelpp Debian repository
RUN apt update \
    && apt install -y lsb-release \
    && apt install -y debian-keyring curl wget gpg sudo  bash-completion tree coreutils \
    # Configure apt and install packages \
    && cp /usr/share/keyrings/debian-maintainers.gpg /etc/apt/trusted.gpg.d \
    && apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    && apt-get -y install git iproute2 procps lsb-release iputils-ping nmap \
    #\
    && apt-get -y install build-essential python3-minimal libpython3-dev \
    && apt-get -y install python-is-python3 python3-pip python3-venv python3-jinja2 \
    && apt-get -y install python3-autopep8 black yapf3 python3-bandit flake8 pydocstyle pylint python3-pytest mypy \
    && apt-get -y install python3-matplotlib \
    && apt-get -y install python3-statsmodels \
    && apt-get -y install python3-pandas python3-seaborn \
    && apt-get -y install python3-numpy python3-scipy \
    && apt-get -y install python3-nlopt python3-tabulate \
    && apt-get -y install python3-pip python-is-python3 \
    && apt-get -y install python3-pint \
    && apt-get -y install python3-decouple python3-requests python3-natsort \
    && apt-get install python3-iapws \
    && apt-get -y install zutty \
    && apt-get -y install texlive-latex-base texlive-latex-extra dvipng 

#set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo,video ${USERNAME} \
    && mkdir -p  /etc/sudoers.d/ \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} \
    && mkdir -p ~${USERNAME}/.ssh/ \
    && ssh-keyscan github.com >> ~${USERNAME}/.ssh/known_hosts \
    && chown -R ${USER_UID}:${USER_GID} ~${USERNAME}/.ssh

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# install visidata in virtual env for $USERNAME

# Define user
USER ${USERNAME}
ENV HOME=/home/${USERNAME}

# set alias, OpenMP Threads and OpenBLAS (threads should be 1 to ensure performance)
RUN echo "alias cp='cp -i'" > $HOME/.bash_aliases && \
    echo "alias egrep='egrep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias fgrep='fgrep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias grep='grep --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias ls='ls --color=auto'" >> $HOME/.bash_aliases && \
    echo "alias mv='mv -i'" >> $HOME/.bash_aliases && \
    echo "alias rm='rm -i'" >> $HOME/.bash_aliases



