import Image
import numpy as np
import ImageFilter


def PIL2array(img):
    """Transfer a PythonImageLibrary (PIL) object into a numpy N x M x (R,G,B) matrix."""
    return np.array(img.getdata(), np.uint16).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    """Transfer a numpy N x M x (R,G,B) matrix into a PythonImageLibrary (PIL) object."""
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def importchan(arr, channel):
	"""Splice an array by channel, make sure that int16."""
	return (np.array(arr[:,:,channel],dtype=np.int16))
    
def normalize_range(color,range=8):
    """Perform image normalization by subtracting mean and spreading over the range of 8bit."""
    dynamic_range = color.max() - color.min()
    maximum_range = 2**range
    return ((color - color.min())*maximum_range/dynamic_range)    
    
def normalize(color):
    """Perform image normalization by subtracting mean and spreading by stddev."""
    return ((color - color.mean())/ color.std())
    
def equalize(im):
    """Perform histogram spread on a given PIL image"""
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(operator.add, h[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut*3)
    
    
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"
    
    def __init__(self, radius=2):
        self.radius = radius
    
    def filter(self, image):
        return image.gaussian_blur(self.radius)


