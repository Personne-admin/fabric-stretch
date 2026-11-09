ARG BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG COWTRACKER_REPO=https://github.com/facebookresearch/cowtracker.git
ARG COWTRACKER_COMMIT=1454f20045d3b514e5b8417907152677f3dba621
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG TRACKER_XFORMERS_SPEC=xformers
ARG TRACKER_PIP_PACKAGES="timm==1.0.25 numpy scipy einops pillow opencv-python-headless huggingface_hub matplotlib"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch, torchvision and xformers are resolved together so xformers matches torch.
RUN pip install "torch==${TORCH_VERSION}" torchvision "${TRACKER_XFORMERS_SPEC}" \
      --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple \
 && pip install ${TRACKER_PIP_PACKAGES}

# The model imports vggt and Depth-Anything-V2 from submodules of this commit.
RUN git clone --quiet "${COWTRACKER_REPO}" /opt/cowtracker \
 && git -C /opt/cowtracker -c advice.detachedHead=false checkout --quiet "${COWTRACKER_COMMIT}" \
 && git -C /opt/cowtracker submodule update --init --recursive --quiet

COPY src/fabsim/cowtracker_to_fabsim.py /opt/fabsim/cowtracker_to_fabsim.py

LABEL org.opencontainers.image.title="fabsim-tracker" \
      org.opencontainers.image.description="CoWTracker tracking step of the fabric-stretch pipeline" \
      org.opencontainers.image.licenses="Apache-2.0"

VOLUME ["/data", "/ckpt"]
WORKDIR /data
ENTRYPOINT ["python", "/opt/fabsim/cowtracker_to_fabsim.py", "--cowtracker-root", "/opt/cowtracker"]
CMD ["--help"]
