import tensorflow as tf

ONNX_MODEL_PATH = "../models/mobilenetv2_cifar10.onnx"
TFLITE_MODEL_PATH = "../models/mobilenetv2_cifar10.tflite"

# Load ONNX via TensorFlow (onnx was already exported)
# TensorFlow reads via SavedModel, so we convert ONNX -> TF -> TFLite
# Use tf.experimental (works for this pipeline)

# Convert ONNX to TensorFlow SavedModel using tf.experimental
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load(ONNX_MODEL_PATH)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# Convert TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables quantization

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite conversion complete.")
