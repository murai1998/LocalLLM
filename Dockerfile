FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    git cmake build-essential libopenblas-dev pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build \
        -DGGML_METAL=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=OFF && \
    cmake --build build --config Release -j$(nproc) && \
    mkdir -p /output/bin /output/lib && \
    cp build/bin/llama-server /output/bin/ && \
    find build -name "*.so*" -exec cp -v {} /output/lib/ \; 2>/dev/null || true && \
    cd .. && rm -rf llama.cpp


FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /output/bin/ /usr/local/bin/
COPY --from=builder /output/lib/ /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"
EXPOSE 8090 8501
CMD ["bash", "-c", "localllm-download --force && localllm-serve"]