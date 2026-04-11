ARG VARIANT=5
FROM mcr.microsoft.com/devcontainers/universal:${VARIANT}

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    PIPX_HOME=/usr/local/pipx \
    PIPX_BIN_DIR=/usr/local/bin \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    CUDA_HOME=/usr/local/cuda \
    UV_LINK_MODE=copy \
    PATH=/usr/local/cargo/bin:/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bat \
        bash-completion \
        dnsutils \
        fd-find \
        fzf \
        htop \
        iputils-ping \
        just \
        lsof \
        netcat-openbsd \
        postgresql-client \
        redis-tools \
        ripgrep \
        rsync \
        shellcheck \
        shfmt \
        sqlite3 \
        strace \
        tmux \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && if [ -x /usr/bin/batcat ]; then ln -sf /usr/bin/batcat /usr/local/bin/bat; fi

RUN arch="$(dpkg --print-architecture)" \
    && case "${arch}" in \
        amd64) cuda_arch="x86_64" ;; \
        arm64) cuda_arch="sbsa" ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && wget -qO /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${cuda_arch}/cuda-keyring_1.1-1_all.deb" \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc

ARG YQ_VERSION=v4.44.3
RUN arch="$(dpkg --print-architecture)" \
    && case "${arch}" in \
        amd64) yq_arch="amd64" ;; \
        arm64) yq_arch="arm64" ;; \
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac \
    && wget -qO /usr/local/bin/yq "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_${yq_arch}" \
    && chmod +x /usr/local/bin/yq

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    && rustup component add clippy rustfmt \
    && chmod -R a+rX /usr/local/cargo /usr/local/rustup

RUN python3 -m pip install --no-cache-dir --break-system-packages pipx \
    && pipx install uv \
    && pipx install ruff \
    && pipx install pre-commit

RUN corepack enable

RUN printf '%s\n' \
    'export EDITOR=vim' \
    'export VISUAL=vim' \
    'export PAGER=less' \
    'export LESS=-FRX' \
    "alias ll='ls -alF'" \
    "alias la='ls -A'" \
    "alias l='ls -CF'" \
    > /etc/profile.d/00-codex-dev.sh

RUN git config --system init.defaultBranch main \
    && git config --system fetch.prune true \
    && git config --system pull.rebase false \
    && git config --system core.autocrlf input

USER codespace
