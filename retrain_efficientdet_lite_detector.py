
import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

"""## Load the training data

The salad dataset is a subset of [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html), and it includes labels to identify five classes:
"Salad", "Seafood", "Tomato", "Baked goods", and "Cheese".

Model Maker requires that we load our dataset using the [`DataLoader`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/DataLoader) API. So in this case, we'll load it from a CSV file that defines 175 images for training, 25 images for validation, and 25 images for testing.
"""

train_data, validation_data, test_data = object_detector.DataLoader.from_csv('./output.csv')

"""If you want to load your own dataset as a CSV file, you can learn more about the format in [Formatting a training data CSV](https://cloud.google.com/vision/automl/object-detection/docs/csv-format). You can load your CSV either from [Cloud Storage](https://cloud.google.com/storage) (as shown above) or from a local path.

[`DataLoader`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/DataLoader) can also load your dataset in other formats, such as from a set of TFRecord files or from a local directory using the PASCAL VOC format.

## Select the model spec

Model Maker supports the EfficientDet-Lite family of object detection models that are compatible with the Edge TPU. (EfficientDet-Lite is derived from [EfficientDet](https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html), which offers state-of-the-art accuracy in a small model size). There are several model sizes you can choose from:

|| Model architecture | Size(MB)* | Latency(ms)** | Average Precision*** |
|-|--------------------|-----------|---------------|----------------------|
|| EfficientDet-Lite0 | 4.4       | 37            | 25.69%               |
|| EfficientDet-Lite1 | 5.8       | 49            | 30.55%               |
|| EfficientDet-Lite2 | 7.2       | 69            | 33.97%               |
|| EfficientDet-Lite3 | 11.4      | 116           | 37.70%               |
|| EfficientDet-Lite4 | 19.9      | 260           | 41.96%               |
| <td colspan=4><br><i>* File size of the integer quantized models. <br/>** Latency measured on Pixel 4 using 4 threads on CPU. <br/>*** Average Precision is the mAP (mean Average Precision) on the COCO 2017 validation dataset.</i></td> |

Beware that the bigger models (Lite3 and Lite4) do not fit onto the Edge TPU's onboard memory, so you'll see even greater latency when using those due to the cost of fetching data from the host system memory. Maybe this extra latency is okay for your application, but if it's not and you require the precision of the larger models, then you can [pipeline the model across multiple Edge TPUs](https://coral.ai/docs/edgetpu/pipeline/) (more about this when we compile the model below).

For this tutorial, we'll use Lite0:
"""

spec = object_detector.EfficientDetLite0Spec()

"""The [`EfficientDetLite0Spec`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/EfficientDetLite0Spec) constructor also supports several arguments that specify training options, such as the max number of detections (default is 25 for the TF Lite model) and whether to use Cloud TPUs for training. You can also use the constructor to specify the number of training epochs and the batch size, but you can also specify those in the next step.

## Create and train the model

Now we need to create our model according to the model spec, load our dataset into the model, specify training parameters, and begin training. 

Using Model Maker, we accomplished all of that with [`create()`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/create):
"""

model = object_detector.create(train_data=train_data, 
                               model_spec=spec, 
                               validation_data=validation_data, 
                               epochs=2, 
                               batch_size=2, 
                               train_whole_model=True)

"""## Evaluate the model

Now we'll use the remaining 25 images in our test dataset to evaluate how well the model performs with data it has never seen before.

The [`evaluate()`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/ObjectDetector#evaluate) method provides output in the style of [COCO evaluation metrics](https://cocodataset.org/#detection-eval):
"""

model.evaluate(test_data)

"""Because the default batch size for [EfficientDetLite models](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/EfficientDetSpec) is 64, this needs only 1 step to go through all 25 images. You can also specify the `batch_size` argument when you call [`evaluate()`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/ObjectDetector#evaluate).

## Export to TensorFlow Lite

Next, we'll export the model to the TensorFlow Lite format. By default, the [`export()`](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/ObjectDetector#export) method performs [full integer post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization), which is exactly what we need for compatibility with the Edge TPU. (Model Maker uses the same dataset we gave to our model spec as a representative dataset, which is required for full-int quantization.)

We just need to specify the export directory and format. By default, it exports to TF Lite, but we also want a labels file, so we declare both:
"""

# m = model.create_model()
# m.compile()
# m.fit()
# m.save('pred_model')

model._export_saved_model('pred_model')

# model.export(export_dir='.',
#              tflite_filename='efficientdet-lite-car-lp-det.tflite',
#              label_filename='lite-car-lp-det-labels.txt',
#              export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

"""### Evaluate the TF Lite model

Exporting the model to TensorFlow Lite can affect the model accuracy, due to the reduced numerical precision from quantization and because the original TensorFlow model uses per-class [non-max supression (NMS)](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH) for post-processing, while the TF Lite model uses global NMS, which is faster but less accurate.

Therefore you should always evaluate the exported TF Lite model and be sure it still meets your requirements:
"""

# model.evaluate_tflite('efficientdet-lite-car-lp-det.tflite', test_data)
