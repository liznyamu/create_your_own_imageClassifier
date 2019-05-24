# Predict flower name from an image with predict.py along with the probability of that name.
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage:
#       python predict.py /path/to/image checkpoint
#       python predict.py flowers/test/93/image_06014.jpg checkpoint.pth

# Options:
# Return top K most likely classes:
#   python predict.py input checkpoint --top_k 3
#   python predict.py flowers/test/93/image_06014.jpg checkpoint.pth --top_k 3
# Use a mapping of categories to real names:
#   python predict.py input checkpoint --category_names cat_to_name.json
#   python predict.py flowers/test/93/image_06014.jpg checkpoint.pth --category_names cat_to_name.json
# Use GPU for inference:
#   python predict.py input checkpoint --gpu
#   python predict.py flowers/test/93/image_06014.jpg checkpoint.pth --gpu


# Imports here
import torch
import numpy as np
from helpers.our_model import (load_checkpoint, predict, predict_device, 
load_testdata, test_model)
from helpers.label_mapping import map_label
import argparse

# handling options and arguments
parser = argparse.ArgumentParser(
    description='Predict Flower class for an input image',
)

parser.add_argument('input', action='store',
                    default = 'flowers/test/93/image_06014.jpg',
                    help='Store path to a single input image')

parser.add_argument('checkpoint', action='store',
                    default='checkpoint.pth',
                    help='Store path to our trained model checkpoint')

parser.add_argument('--top_k', action='store', type = int, dest = 'top_k',
                    default = 1, help='Store K top classes')

parser.add_argument('--category_names', action='store', dest = 'category_names',
                    help='Store path to Flower category names')

parser.add_argument('--gpu', action='store_true',
                    default = False, dest = 'gpu',
                    help='Store value indicating whether or not use GPU in inference')

# argparser docs - http://bit.ly/2M7BWg5
results = parser.parse_args()


# path to the single image to predict a flower class
image_path = results.input      # 'flowers/test/93/image_06014.jpg'

# path to our trained model checkpoint
checkpoint_path = results.checkpoint    #'vgg19_woM_checkpoint.pth'

# number of top K classes, and probabilities to return
class_count = results.top_k  # 1

# mapping of categories to real names
map_path = results.category_names #'cat_to_name.json'

# use GPU for inference
gpu = results.gpu #False

# load the trained model and checkpoint
trained_model, checkpoint = load_checkpoint(checkpoint_path)

# predict K flower name and probability of that name
#probs, classes = predict(image_path, trained_model, topk=class_count)
probs, classes = predict_device(image_path, trained_model, gpu, topk=class_count)

if map_path is None:
    # Option 1 : K classes and probabilities
    print('\nThe model\'s top predictions : ', ' ... '*3)
    print('Flower Class(es): ', classes)
    print('Probability(ies): ', probs, '\n')
else:
    # Option 2 : K actual classes and probabilities
    # convert from the class integer encoding to actual
    # K flower names with the cat_to_name.json
    cat_to_name = map_label(map_path)
    flower_class = [cat_to_name.get(flower) for flower in classes]
    # Predicted flower names and their probabilities
    print('\nThe model\'s top predictions : ', ' ... '*3)
    print('Flower Category(ies): ', flower_class)
    print('Probability(ies): ', probs, '\n')
