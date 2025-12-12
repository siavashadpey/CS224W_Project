FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN conda config --set ssl_verify false && \
    conda clean -afy

WORKDIR /workspace

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host data.pyg.org \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torchviz \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    jupyterlab \
    notebook \
    tensorboard \
    numpy \
    scikit-learn \
    matplotlib \
    pyyaml \
    requests \
    tqdm \
    pytest \
    jupyter \
    google-cloud-storage \
    google-cloud-aiplatform \
    wandb \
    tensorboard \ 
    cloudml-hypertune

COPY . .
RUN pip install --no-cache-dir -e .

RUN mkdir -p /workspace/checkpoints /workspace/data /workspace/logs

EXPOSE 6006

ENTRYPOINT ["python", "scripts/train_masked_autoencoder.py"]
