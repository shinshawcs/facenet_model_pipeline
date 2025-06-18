FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy script and facenet_pytorch weights if needed
COPY . .  

# Set entrypoint
CMD ["python", "facenet_evaluate.py"]