# Turing Engine serving image: `turing serve` on port 8000.
#   CPU  (linux/amd64, linux/arm64): docker build -f deploy/Dockerfile -t turing:cpu .
#   CUDA (linux/amd64):              docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t turing:cuda .
# The CUDA build differs only in its PyTorch wheels (which bundle the CUDA runtime); the host needs the
# NVIDIA driver and Container Toolkit. See docs/docker.md.

FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN pip wheel --no-cache-dir --no-deps -w /wheels /src

FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY --from=build /wheels /wheels
# the llmd extra (pyzmq, msgpack) is what `turing serve --enable-kv-events` publishes with
RUN pip install --no-cache-dir "$(ls /wheels/*.whl)[llmd]" && rm -rf /wheels \
    && useradd --create-home turing
USER turing
WORKDIR /home/turing
EXPOSE 8000
ENTRYPOINT ["turing", "serve", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--model", "smollm2"]
