import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            output_tensor = pb_utils.get_input_tensor_by_name(request, "output").as_numpy()
            print("Received shape:", output_tensor.shape) 
            probs = softmax(output_tensor)
            top_classes = np.argmax(probs, axis=1).astype(np.int32)

            out_tensor = pb_utils.Tensor("category", top_classes)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

    def finalize(self):
        pass

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)