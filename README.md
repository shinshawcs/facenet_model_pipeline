Facenet-Based Face Recognition Pipeline

This project implements a full MLOps pipeline for facial identification using InceptionResnetV1. The core functionality is to extract face embeddings and support identity matching. The pipeline supports end-to-end deployment from data preprocessing to optimized GPU inference in production.

Technologies Used:
- Model: InceptionResnetV1 (from facenet-pytorch)
- Quantization: QAT (Quantization-Aware Training)
- Inference Runtime: TensorRT (FP16)
- Pipeline Orchestration: Apache Airflow + KubernetesPodOperator
- Serving: FastAPI (REST API endpoint)
- Deployment: GCP Kubernetes Engine (GKE), Docker
- Monitoring: Prometheus + Grafana

Project Structure
├── airflow_pipeline/
├── data_preprocessing/
├── model_finetuning/
├── model_quantization/
├── model_compilation/
├── model_evaluation/ 
├── model_benchmark/
├── triton_models/
├── fastapi_endpoints/

Pipeline Stages
1.	Data Preprocessing
	•	Download LFW
	•	Apply MTCNN face alignment (160x160)
	•	Save into ImageFolder structure
2.	Model Finetuning & Quantization
	•	Fine-tune InceptionResnetV1 on aligned dataset
	•	Apply QAT for better accuracy
3.	Compilation
	•	Export to ONNX
	•	Compile with TensorRT (FP16 engine)
4.	Evaluation & Benchmarking
	•	Measure latency (ONNX Runtime vs TensorRT)
	•	Validate accuracy using test embeddings
5.	Model Deployment
	•	FastAPI endpoint to receive face image
	•	Output 512-dim embedding for downstream ID match

FastAPI Endpoint (Example)
POST /inference
{
    "username": "<username>",
    "image_url": "<url_of_face_image>"
}
{
    "status": "success",
    "request_id": 42,
    "username": "alice",
    "image_url": "https://storage.googleapis.com/your-bucket/folder/image.jpg",
    "prediction": "John_Doe"
}








Deployment