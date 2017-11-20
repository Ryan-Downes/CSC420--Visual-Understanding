from os.path import isfile
import cv2 as cv
import numpy as np
import dataset as ds

def read_image(filename):
    success = False
    msg = ''
    images = cv.imread(filename, 0)
    if images is None:
        if not isfile(filename):
            msg = filename + ' can not be found.'
        else:
            msg = 'The file ' + filename + ' can be found but not read.'
    else:
        success = True

    return success, images, msg


def write_image(self, filename, key):
    success = False
    msg = 'No Image Available'

    if self._images[key] is None:
        msg = 'Unable to save empty image to ' + filename + ' .'

    else:
        cv.imwrite(filename, self._images[key])
        success = True
    return success, msg
if __name__ == '__main__':

    dataset = ds.read_dataset()
    cv.imshow('image',dataset[1])
    cv.waitKey(0)
    cv.destroyAllWindows()

    exit(1)
