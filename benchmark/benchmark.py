import time
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
def benchmark_onnxruntime(session, input_name, input_data, repeat=100):
    times = []
    for _ in range(repeat):
        start = time.time()
        session.run(None, {input_name: input_data})
        end = time.time()
        times.append((end - start) * 1000)
    return np.mean(times), session.run(None, {input_name: input_data})[0]

def load_trt_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        # Handle dynamic shapes by using a reasonable size
        if -1 in shape:
            shape = [1 if dim == -1 else dim for dim in shape]
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        
        # Print memory allocation info for debugging
        print(f"Allocating buffer for {binding}: shape={shape}, size={size}, dtype={dtype}")
        
        try:
            # Try to allocate page-locked memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
        except cuda.MemoryError:
            # Fallback to regular memory if page-locked allocation fails
            print(f"⚠️ Warning: Failed to allocate page-locked memory for {binding}, using regular memory")
            host_mem = np.empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))
    return inputs, outputs, bindings

def infer_trt(engine, context, inputs, outputs, bindings, dummy_input,repeat=100):
    stream = cuda.Stream()
    in_host, in_dev = inputs[0]
    out_host, out_dev = outputs[0]

    # Set dynamic dimensions for input bindings only
    for binding_idx in range(engine.num_bindings):
        binding = engine[binding_idx]
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            shape = engine.get_tensor_shape(binding)
            
            # Check if this binding has dynamic dimensions
            if any(dim == -1 for dim in shape):
                # Get the actual shape from dummy_input
                actual_shape = dummy_input.shape
                print(f"Setting dynamic shape for input {binding}: {actual_shape}")
                context.set_input_shape(binding, actual_shape)

    in_host[:] = dummy_input.ravel()

    start_event = cuda.Event()
    end_event = cuda.Event()
    times = []
    for _ in range(repeat):
        start_event.record(stream)
        cuda.memcpy_htod_async(in_dev, in_host, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(out_host, out_dev, stream)
        end_event.record(stream)
        stream.synchronize()
        times.append(start_event.time_till(end_event))

    # Get the actual output shape from the context for the output binding
    output_binding = engine[1]   # Get the output binding index
    output_shape = context.get_tensor_shape(output_binding)
    print(f"Output shape after inference: {output_shape}")
    output = out_host.reshape(output_shape)
    return np.mean(times), output

def main(**context):
    base_dir = Path(os.environ.get('MODEL_BASE_PATH', '/opt/airflow/shared/models'))
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under {base_dir}")
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    results = []
    dummy_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
    print("\n🧪 Begin (Benchmark)...")
    print("-" * 60)
    print(f"{'Model':20s} {'Latency(ms)':>12s} {'MAE vs FP32':>16s} {'Note':>20s}")

    # ONNX FP32
    print("🔄 Running ONNX Runtime (Non QAT FP32) on GPU")
    onnx_fp32_path = f"{latest_subdir}/facenet_model_fp32.onnx"
    if not os.path.exists(onnx_fp32_path):
        print(f"❌ ONNX FP32 Not Exist: {onnx_fp32_path}")
        return
    try:
        ort_session = ort.InferenceSession(onnx_fp32_path, providers=["CUDAExecutionProvider"])
        ort_input_name_fp32 = ort_session.get_inputs()[0].name
        onnx_time_fp32, onnx_output_fp32 = benchmark_onnxruntime(ort_session, ort_input_name_fp32, dummy_input)
        results.append(("ONNX FP32", onnx_time_fp32, onnx_output_fp32))
        print(f"{'ONNX FP32':20s} {onnx_time_fp32:12.3f} {'-':>16s} {'(baseline)':>20s}")
    except Exception as e:
        print(f"❌ ONNX FP32 Failed: {e}")
        return
    # TensorRT FP16
    print("🔄 Running TensorRT (Non QAT FP16) on GPU")
    trt_fp32_path = f"{latest_subdir}/facenet_model_fp16.trt"
    if not os.path.exists(trt_fp32_path):
        print(f"❌ TensorRT FP32 engine 不存在: {trt_fp32_path}")
        return
    try:
        trt_engine_fp32 = load_trt_engine(trt_fp32_path)
        trt_context_fp32 = trt_engine_fp32.create_execution_context()
        trt_inputs_fp32, trt_outputs_fp32, trt_bindings_fp32 = allocate_buffers(trt_engine_fp32)
        trt_time_fp32, trt_output_fp32 = infer_trt(
            trt_engine_fp32, trt_context_fp32, trt_inputs_fp32, trt_outputs_fp32, trt_bindings_fp32,dummy_input=dummy_input
        )
        results.append(("TensorRT FP16", trt_time_fp32, trt_output_fp32))
    except Exception as e:
        print(f"❌ TensorRT FP16 推理失败: {e}")
        return
    # TensorRT INT8 (QAT)
    print("🔄 Running TensorRT (QAT INT8) on GPU")
    trt_int8_path = f"{latest_subdir}/facenet_model_qat_int8.trt"
    if not os.path.exists(trt_int8_path):
        print(f"❌ TensorRT INT8 engine 不存在: {trt_int8_path}")
        return
    try:
        trt_engine_int8 = load_trt_engine(trt_int8_path)
        trt_context_int8 = trt_engine_int8.create_execution_context()
        trt_inputs_int8, trt_outputs_int8, trt_bindings_int8 = allocate_buffers(trt_engine_int8)
        trt_time_int8, trt_output_int8 = infer_trt(
            trt_engine_int8, trt_context_int8, trt_inputs_int8, trt_outputs_int8, trt_bindings_int8,
            dummy_input=dummy_input
        )
        results.append(("TensorRT INT8", trt_time_int8, trt_output_int8))
    except Exception as e:
        print(f"❌ TensorRT INT8 推理失败: {e}")
        return
    
    print("\n🧪 Benchmark Summary:")
    baseline_output = results[0][2].astype(np.float32)
    for name, time_ms, output in results:
        note = ""
        if output.shape != baseline_output.shape:
            note = f"⚠️ Shape mismatch: {output.shape} vs {baseline_output.shape}"
        elif np.isnan(output).any():
            note = "⚠️ Output contains NaNs"
        elif name == "ONNX FP32":
            note = "(baseline)"
        else:
            mae = np.mean(np.abs(output.astype(np.float32) - baseline_output))
            note = f"{mae:.6f}"
        print(f"{name:20s} {time_ms:12.3f} {('-' if name=='ONNX FP32' else note):>16s} {note:>20s}")
        
    latency_path = f"{base_dir}/latency.txt"
    with open(latency_path, "w") as f:
        f.write(f"{trt_time_int8:.4f}")
    print(f"\n⏱️ Benchmark latency written to {latency_path}")
    return results

if __name__ == "__main__":
    main()