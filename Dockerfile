FROM python:3.11.4-bullseye@sha256:4b3c9c338fdf1db596eb1ccf83597b879098aecf30479a9f01839ab1f1cf0772

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV UV_VERSION="0.10.12"
ENV POETRY_VERSION="2.1.1"
ENV POETRY_SHA256="1d433880bd5b401327ddee789ccfe9ff197bf3b0cd240f0bc7cc99c84d14b16c"
ENV UV_SHA256_AMD64="101481a1f48db6becf219914a591a588c0b3bfd05bef90768a5d04972bd6455e"
ENV UV_SHA256_ARM64="a5afe619e8a861fe4d49df8e10d2c6963de0dac6b79350c4832bf3366c8496cf"
ENV PROTOBUF_VERSION="33.1"
ENV PROTOBUF_SHA256="f3340e28a83d1c637d8bafdeed92b9f7db6a384c26bca880a6e5217b40a4328b"

# Install tooling and protoc, then clean up build deps
RUN apt-get -qq update && apt-get -qq -y install curl vim zip unzip htop\
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
        amd64) uv_wheel="uv-${UV_VERSION}-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"; uv_sha256="$UV_SHA256_AMD64" ;; \
        arm64) uv_wheel="uv-${UV_VERSION}-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl"; uv_sha256="$UV_SHA256_ARM64" ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac \
    && poetry_wheel="poetry-${POETRY_VERSION}-py3-none-any.whl" \
    && curl -LO "https://files.pythonhosted.org/packages/py3/u/uv/${uv_wheel}" \
    && curl -LO "https://files.pythonhosted.org/packages/py3/p/poetry/${poetry_wheel}" \
    && echo "${uv_sha256}  ${uv_wheel}" | sha256sum --check --strict \
    && echo "${POETRY_SHA256}  ${poetry_wheel}" | sha256sum --check --strict \
    && python3 -m pip install --no-cache-dir "./${uv_wheel}" "./${poetry_wheel}" \
    && rm -f "${uv_wheel}" "${poetry_wheel}" \
    && poetry config virtualenvs.create false \
    && PB_REL="https://github.com/protocolbuffers/protobuf/releases" \
    && curl -LO $PB_REL/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && echo "${PROTOBUF_SHA256}  protoc-${PROTOBUF_VERSION}-linux-x86_64.zip" | sha256sum --check --strict \
    && unzip protoc-${PROTOBUF_VERSION}-linux-x86_64.zip -d $HOME/.local \
    && rm protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && apt-get -qq -y remove curl unzip \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

WORKDIR /app
