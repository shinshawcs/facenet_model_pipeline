# scripts/quantize_dynamic.py

import torch
import os
from facenet_pytorch import InceptionResnetV1

torch.backends.quantized.engine = 'fbgemm' 
def main():
    model_path = '/home/egatech2017/airflow/models/facenet_model.pth'
    output_path = '/home/egatech2017/airflow/models/facenet_model_dynamic_quantized.pth'

    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # load original model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(checkpoint['classes'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # only Linear quantization
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    print(model_quantized)
    # save quantized model
    torch.save({
        'model_state_dict': model_quantized.state_dict(),
        'classes': checkpoint['classes']
    }, output_path)

    print(f"✅ Dynamic quantized model saved to {output_path}")

    dummy_input = torch.randn(1, 3, 160, 160)
    torch.onnx.export(
        model,
        dummy_input,
        "/home/egatech2017/airflow/models/facenet_model_dynamic_quantized.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )
    print("✅ ONNX exported to models/facenet_model_dynamic_quantized.onnx")

if __name__ == "__main__":
    main()