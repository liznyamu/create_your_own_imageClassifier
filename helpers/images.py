# Import to diplay images
from PIL import Image
import numpy as np

# TODO: Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    pil_image = Image.open(image_path)

    # resize the image to 256 pixels using PIL and maintain its aspect ratio - http://bit.ly/2M1G558
    size = 256, 256
    pil_image.thumbnail(size, Image.ANTIALIAS)


    # crop the center 224x224 portion of the image - http://bit.ly/2Qic4ww
    w, h = pil_image.width, pil_image.height
    left, upper = w//2 - 224//2, h//2 - 224//2
    right, lower = w//2 + 224//2, h//2 + 224//2
    pil_image = pil_image.crop((left, upper, right, lower))
    #imshow(np.asarray(pil_image)) # view image on jupyter - http://bit.ly/2wc7Mxh

    # convert the color channels of the image- divide by 255 to get floats between 0-1
    np_image = np.array(pil_image)/255

    # normalize the images -
    # subtract the means from each color channel, then divide by the standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_image = (np_image - mean) / std

    # reorder dimensions: - http://bit.ly/2Qh9SoS
    # color channel is the 3rd dimension
    # color channel needs to be 1st and retain the order of the other two dimensions.
    norm_image = norm_image.transpose((2, 0, 1))

    #return processed image
    return norm_image
