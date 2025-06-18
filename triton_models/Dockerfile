
FROM nvcr.io/nvidia/tritonserver:25.02-py3

WORKDIR /models
RUN apt-get update && \
    apt-get install -y curl && \
    curl https://sdk.cloud.google.com | bash && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade google-auth google-cloud-storage Pillow

# Expose Triton ports
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Run Triton Server with GCS path
CMD ["tritonserver", "--model-repository=gs://my-facenet-bucket/triton/models", "--log-verbose=1"]
