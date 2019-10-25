import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import os
from PIL import Image
import numpy as np
import imageio


def representative_dataset_gen():
  for _ in range(100):
      img = imageio.imread('cat.jpg', pilmode='RGB')
      img = np.array(Image.fromarray(img).resize((120, 120))).astype(np.float32)
      img[:, :, 0] -= 123.68
      img[:, :, 1] -= 116.779
      img[:, :, 2] -= 103.939
      img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
      data = np.expand_dims(img, axis=0)
      yield [data]

model = tf.keras.models.load_model("hccr_googlenet_keras.h5")
model.summary()

# converting currently with OPTIMIZE FOR SIZE argument
# With snippets of sample data, will be able to call a full model (weights and other ops) quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
print("Converting and Quantizing...")
tflite_model = converter.convert()

print("Saving tflite file...")
open("hccr_googlenet_lite.tflite", "wb").write(tflite_model)
