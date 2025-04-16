import torch
import time
from facenet_pytorch import InceptionResnetV1

torch.backends.quantized.engine = 'fbgemm'
def benchmark_latency_main(**context):
    model_path = "/home/egatech2017/airflow/models/facenet_model_dynamic_quantized.pth"

    checkpoint = torch.load(model_path, map_location='cpu')
    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(checkpoint['classes'])
    )
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    model_quantized.load_state_dict(checkpoint['model_state_dict'])
    model_quantized.eval()

    input_tensor = torch.randn(1, 3, 160, 160)  # input tensor
    warmup = 10
    repeat = 100

    # Warmup
    for _ in range(warmup):
        _ = model_quantized(input_tensor)
    
    start = time.time()
    for _ in range(repeat):
        _ = model_quantized(input_tensor)
    end = time.time()

    avg_latency = (end - start) / repeat
    print(f"⚡️ Average Inference Latency: {avg_latency * 1000:.2f} ms")
    ti = context.get("ti")
    print("Context:", context)
    print("ti: ", ti)
    if ti is not None:
        ti.xcom_push(key="latency", value=avg_latency)
    with open("/home/egatech2017/airflow/results/latency.txt", "w") as f:
        f.write(str(avg_latency))
    return avg_latency

if __name__ == "__main__":
    benchmark_latency_main()