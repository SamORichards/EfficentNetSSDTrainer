
# Set the model files
MODEL_FILE = 'efficientdet-lite-car-lp.tflite'
LABELS_FILE = 'lite-car-lp-det-labels.txt'
DETECTION_THRESHOLD = 0.4

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import tflite_runtime.interpreter as tflite 
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file

def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)
  for obj in objs:
    bbox = obj.bbox
    color = tuple(int(c) for c in COLORS[obj.id])
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline=color, width=15)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=90)
    draw.text((bbox.xmin + 20, bbox.ymin + 20),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color, font=font)

# Load the TF Lite model
labels = read_label_file(LABELS_FILE)
interpreter = tflite.Interpreter(MODEL_FILE)
interpreter.allocate_tensors()

# Resize the image
image = Image.open(INPUT_IMAGE)
_, scale = common.set_resized_input(
    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

# Run inference and draw boxes
interpreter.invoke()
objs = detect.get_objects(interpreter, DETECTION_THRESHOLD, scale)
draw_objects(ImageDraw.Draw(image), objs, labels)

# Show the results
width = 400
height_ratio = image.height / image.width
image.resize((width, int(width * height_ratio)))