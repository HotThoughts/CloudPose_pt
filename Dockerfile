ARG BASE_IMAGE_NAME=pytorchlightning/pytorch_lightning
ARG BASE_IMAGE_TAG=latest-py3.8-torch1.5
FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

RUN apt-get update -qq \
   && apt-get install -y --no-install-recommends \

   # Cleanup
   && apt-get autoremove -y \
   && rm -rf /var/lib/apt/lists/*

# change working directory
WORKDIR /build

# install dependencies
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate && \
  pip install --no-cache-dir -U "pip==21.*"

# ENTRYPOINT python
# CMD train.py --batch_size 128 --log_dir log --num_point 256
