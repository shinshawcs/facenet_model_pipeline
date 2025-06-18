FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY . .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \
    USE_WANDB=false

CMD ["python", "facenet_finetune.py"]