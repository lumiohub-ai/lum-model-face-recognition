# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv

# ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04@sha256:131e238d724ee145317f10d6c8eba0d301439c6c8764b02473510e7035756e81

ARG DEBIAN_FRONTEND=noninteractive
ARG FR_SLUG="face-recognition"


## ============================================================================
## STAGE 1: Builder - Install Python dependencies (cached layer)
## ============================================================================
FROM ${BASE_IMAGE} AS builder

ARG DEBIAN_FRONTEND
ARG FR_SLUG
ARG PYTHON_VERSION=3.10

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR "/usr/src/${FR_SLUG}"

# Install system dependencies and Miniconda
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=private \
	--mount=type=cache,target=/root/.cache,sharing=locked \
	rm -rfv /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* && \
	apt-get clean -y && \
	apt-get update --fix-missing -o Acquire::CompressionTypes::Order::=gz && \
	apt-get install -y --no-install-recommends \
	ca-certificates \
	build-essential \
	wget \
	curl \
	nano \
	software-properties-common \
	tzdata && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update --fix-missing -o Acquire::CompressionTypes::Order::=gz && \
	_MINICONDA_VERSION=py310_25.1.1-2 && \
	_MINICONDA_FILENAME=Miniconda3-${_MINICONDA_VERSION}-Linux-x86_64.sh && \
	export _MINICONDA_URL=https://repo.anaconda.com/miniconda/${_MINICONDA_FILENAME}; \
	if [ ! -f "/root/.cache/${_MINICONDA_FILENAME}" ]; then \
		wget -nv --show-progress --progress=bar:force:noscroll "${_MINICONDA_URL}" -O "/root/.cache/${_MINICONDA_FILENAME}"; \
	fi && \
	/bin/bash "/root/.cache/${_MINICONDA_FILENAME}" -b -u -p /opt/conda && \
	/opt/conda/condabin/conda update -y conda && \
	/opt/conda/condabin/conda install -y python=${PYTHON_VERSION} pip && \
	/opt/conda/bin/pip install --timeout 60 -U pip

# Fix TLS in the Conda env
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    /opt/conda/condabin/conda install -y \
        ca-certificates \
        openssl \
        certifi && \
    /opt/conda/condabin/conda update -y ca-certificates openssl && \
    /opt/conda/bin/python -c "import ssl,certifi; print('OpenSSL:', ssl.OPENSSL_VERSION); print('certifi:', certifi.where())"

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Install PyTorch (large, cached separately)
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    /opt/conda/bin/pip install --timeout 1200 --retries 5 \
        torch torchvision --extra-index-url https://download.pytorch.org/whl/cu122

# Copy ONLY dependency files first (for better caching)
COPY requirements.txt ./

# Copy modules for installation (these rarely change)
COPY modules ./modules

# Install modules and requirements (cached unless requirements.txt or modules change)
RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    /opt/conda/bin/pip install --timeout 1200 --retries 5 ./modules/insightface && \
    /opt/conda/bin/pip install --timeout 1200 --retries 5 ./modules/yolo_tracking && \
    /opt/conda/bin/pip install --timeout 1200 --retries 5 -r ./requirements.txt

# Copy setup files and install package (without source - just metadata)
COPY setup.py setup.cfg pyproject.toml ./
COPY src/face_recognition/__version__.py ./src/face_recognition/__version__.py
COPY src/face_recognition/__init__.py ./src/face_recognition/__init__.py

# Create minimal package structure for editable install
RUN mkdir -p ./src/face_recognition && \
    mkdir -p ./src/person_tracking && \
    touch ./src/__init__.py && \
    touch ./src/person_tracking/__init__.py


## ============================================================================
## STAGE 2: Base runtime image
## ============================================================================
FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND
ARG FR_SLUG

ARG FR_HOME_DIR="/app"
ARG FR_DIR="${FR_HOME_DIR}/${FR_SLUG}"
ARG FR_DATA_DIR="/var/lib/${FR_SLUG}"
ARG FR_LOGS_DIR="/var/log/${FR_SLUG}"
ARG FR_TMP_DIR="/tmp/${FR_SLUG}"
ARG FR_MODELS_DIR="${FR_DATA_DIR}/models"
ARG FR_PORT=8000
ARG HASH_PASSWORD="\$5\$II24BPGFtWgt4mLn\$wF09aZOSlBD.ggjwxCxD4kHIbCD/IkRmWTqrKgxG96D"
ARG UID=1000
ARG GID=11000
ARG USER=fr-user
ARG GROUP=fr-group

ENV FR_HOME_DIR="${FR_HOME_DIR}" \
	FR_DIR="${FR_DIR}" \
	FR_DATA_DIR="${FR_DATA_DIR}" \
	FR_LOGS_DIR="${FR_LOGS_DIR}" \
	FR_TMP_DIR="${FR_TMP_DIR}" \
	FR_PORT=${FR_PORT} \
	UID=${UID} \
	GID=${GID} \
	USER=${USER} \
	GROUP=${GROUP} \
	PYTHONIOENCODING=utf-8 \
	PYTHONUNBUFFERED=1 \
	PATH="/opt/conda/bin:${PATH}"

ENV FR_MODELS_DIR="${FR_MODELS_DIR}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN rm -rfv /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /root/.cache/* && \
	apt-get clean -y && \
	apt-get update --fix-missing -o Acquire::CompressionTypes::Order::=gz && \
	apt-get install -y --no-install-recommends \
		ca-certificates \
		openssl \
		sudo \
		gosu \
		locales \
		tzdata \
		procps \
		iputils-ping \
		iproute2 \
		curl \
		nano \
		libgl1-mesa-glx \
		libglib2.0-0 \
		libxvidcore4 && \
	update-ca-certificates && \
	apt-get clean -y && \
	sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
	sed -i -e 's/# en_AU.UTF-8 UTF-8/en_AU.UTF-8 UTF-8/' /etc/locale.gen && \
	sed -i -e 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
	dpkg-reconfigure --frontend=noninteractive locales && \
	update-locale LANG=en_US.UTF-8 && \
	echo "LANGUAGE=en_US.UTF-8" >> /etc/default/locale && \
	echo "LC_ALL=en_US.UTF-8" >> /etc/default/locale && \
	addgroup --gid ${GID} ${GROUP} && \
	useradd -lmN -d "/home/${USER}" -s /bin/bash -g ${GROUP} -G sudo -u ${UID} ${USER} && \
	echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" && \
	chmod 0440 "/etc/sudoers.d/${USER}" && \
	echo -e "${USER}:${HASH_PASSWORD}" | chpasswd -e && \
	echo -e "\nalias ls='ls -aF --group-directories-first --color=auto'" >> /root/.bashrc && \
	echo -e "alias ll='ls -alhF --group-directories-first --color=auto'\n" >> /root/.bashrc && \
	echo -e "\numask 0002" >> "/home/${USER}/.bashrc" && \
	echo "alias ls='ls -aF --group-directories-first --color=auto'" >> "/home/${USER}/.bashrc" && \
	echo -e "alias ll='ls -alhF --group-directories-first --color=auto'\n" >> "/home/${USER}/.bashrc" && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> "/home/${USER}/.bashrc" && \
	echo "conda activate base" >> "/home/${USER}/.bashrc" && \
	rm -rfv /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /root/.cache/* "/home/${USER}/.cache/*" && \
	mkdir -pv "${FR_DIR}" "${FR_DATA_DIR}" "${FR_LOGS_DIR}" "${FR_TMP_DIR}" && \
	chown -Rc "${USER}:${GROUP}" "${FR_HOME_DIR}" "${FR_DATA_DIR}" "${FR_LOGS_DIR}" "${FR_TMP_DIR}" && \
	find "${FR_DIR}" "${FR_DATA_DIR}" -type d -exec chmod -c 770 {} + && \
	find "${FR_DIR}" "${FR_DATA_DIR}" -type d -exec chmod -c ug+s {} + && \
	find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type d -exec chmod -c 775 {} + && \
	find "${FR_LOGS_DIR}" "${FR_TMP_DIR}" -type d -exec chmod -c +s {} +

ENV LANG=en_US.UTF-8 \
	LANGUAGE=en_US.UTF-8 \
	LC_ALL=en_US.UTF-8

# SSL configuration for runtime (needed for InsightFace model downloads)
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Copy conda environment from builder (this is the heavy cached layer)
COPY --from=builder --chown=${UID}:${GID} /opt/conda /opt/conda


## ============================================================================
## STAGE 3: Final application image
## ============================================================================
FROM base AS app

WORKDIR "${FR_DIR}"

# Download ReID model weights (cached layer - rarely changes)
ARG REID_WEIGHTS_URL="https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
ARG REID_WEIGHTS_PATH="volumes/models/weights/osnet_x0_25_msmt17.pt"
RUN mkdir -p "$(dirname ${REID_WEIGHTS_PATH})" && \
    curl -fSL --retry 3 --retry-delay 5 -o "${REID_WEIGHTS_PATH}" "${REID_WEIGHTS_URL}" && \
    chmod 644 "${REID_WEIGHTS_PATH}"

# Copy entrypoint scripts first (rarely change)
COPY --chown=${UID}:${GID} --chmod=770 ./scripts/docker/*.sh /usr/local/bin/

# Copy config files (change occasionally)
COPY --chown=${UID}:${GID} ./configs ${FR_DIR}/configs
COPY --chown=${UID}:${GID} ./.env.example ${FR_DIR}/.env.example

# Copy source code last (changes most frequently)
# NOTE: In development, this is overridden by volume mount in compose.yml
COPY --chown=${UID}:${GID} ./src ${FR_DIR}/src

EXPOSE ${FR_PORT}

# Run entrypoint as root to fix permissions, then switch to fr-user
ENTRYPOINT ["docker-entrypoint.sh"]
