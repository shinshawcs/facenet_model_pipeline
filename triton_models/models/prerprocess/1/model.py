import subprocess
subprocess.call(["/bin/bash", "/models/preprocess/1/install.sh"]) 
import numpy as np
from PIL import Image
import io
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

    def initialize(self, args):
        pass
    def execute(self, requests):
        responses = []
        batch_images = [] 
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            img_bytes = in_tensor.as_numpy().tobytes()

            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img = img.resize((160, 160))

            img_array = np.asarray(img).astype(np.float32) / 255.0 
            img_array = img_array.transpose(2, 0, 1)  
            #img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(img_array)
        
        batch_array = np.stack(batch_images, axis=0)
        
        out_tensor = pb_utils.Tensor("input", batch_array)
        responses = []
        for _ in requests:
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
        # inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
        # return [inference_response]
        # for _ in requests:
        #     out_tensor = pb_utils.Tensor("input", batch_array)
        #     inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
        #     responses.append(inference_response)
        # return responses

    def finalize(self):
        pass