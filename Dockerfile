FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Configure CUDA library paths for runtime (let NVIDIA runtime inject real libs)
RUN ldconfig

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies with CUDA support
COPY requirements.txt .
RUN pip install -r requirements.txt

# Add huggingface CLI to PATH
ENV PATH="/root/.local/bin:$PATH"

# Download model using Hugging Face CLI
RUN hf download meetkai/functionary-small-v3.2-GGUF --local-dir ./model

# Create temporary stub links for compilation
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 && \
    ln -sf /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/libnvidia-ml.so.1 && \
    ldconfig

# Compile llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64" \
    pip install llama-cpp-python[server]

# Remove temporary stub links after compilation
RUN rm -f /usr/local/cuda/lib64/libcuda.so.1 /usr/local/cuda/lib64/libnvidia-ml.so.1 && \
    ldconfig

# Copy application files
COPY test.py .

# Expose port
EXPOSE 8000

# Command to run the server
CMD ["python", "-m", "llama_cpp.server", \
     "--model", "./model/functionary-small-v3.2.Q8_0.gguf", \
     "--chat_format", "functionary-v2", \
     "--hf_pretrained_model_name_or_path", "./model/", \
     "--n_ctx", "24576", \
     "--n_gpu_layers", "-1", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
