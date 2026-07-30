FROM ghcr.io/astral-sh/uv:0.11.30@sha256:93b61e21202b1dab861092748e46bbd6e0e41dd84f59b9174efd2353186e1b47 AS uv

FROM python:3.13.6-bullseye@sha256:f58f33e0563f2ba81c7afe6259cd912f0c33413da93c75cc3a70a941c17afa8c

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV PROTOBUF_VERSION="33.1"
ENV PROTOBUF_SHA256="f3340e28a83d1c637d8bafdeed92b9f7db6a384c26bca880a6e5217b40a4328b"

COPY --from=uv /uv /usr/local/bin/uv

# Install tooling and protoc, then clean up build deps
RUN apt-get -qq update && apt-get -qq -y install curl vim zip unzip htop\
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
