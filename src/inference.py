import onnxruntime as ort
from preprocess import preprocess
import numpy as np

class InferenceModel:
    def __init__(self, model_path, input_size=320):
        self.model_path = model_path
        self.input_size = input_size
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame):
        input_tensor = preprocess(frame, self.input_size)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return np.squeeze(outputs[0])
