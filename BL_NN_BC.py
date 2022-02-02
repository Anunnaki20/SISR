import numpy
import time
from tabulate import tabulate

# Original
import skimage.io
import skimage.transform
import skimage.color

# Lowest Quality
from PIL import Image

# Currently testing
import cv2

# -------------------------------- Image upscalling using skimage ------------------------------

def gray_12bit_ski(image):
    gtImage = skimage.img_as_float(skimage.io.imread(image))
    gtImage = skimage.img_as_uint(skimage.color.rgb2gray(skimage.color.rgba2rgb(gtImage))) & 0xFFF0
    gtImage = skimage.img_as_float(gtImage)
    return gtImage

def NN_ski(image, scale):
    gtImage = gray_12bit_ski(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale
    nnImage = skimage.transform.resize(gtImage, (inRows, inCols), order=0)
    saveFileName = "output/NN_ski.png"
    skimage.io.imsave(saveFileName, nnImage)

def BL_ski(image, scale):
    gtImage = gray_12bit_ski(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale
    blImage = skimage.transform.resize(gtImage, (inRows, inCols), order=1)
    saveFileName = "output/BL_ski.png"
    skimage.io.imsave(saveFileName, blImage)

def BC_ski(image, scale):
    gtImage = gray_12bit_ski(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale
    bcImage = skimage.transform.resize(gtImage, (inRows, inCols), order=3)
    saveFileName = "output/BC_ski.png"
    skimage.io.imsave(saveFileName, bcImage)

# ----------------------------------------------------------------------------------------------


# -------------------------------- Image upscalling using OpenCV2 ------------------------------

def gray_12bit_cv(image):
    gtImage = cv2.imread(image).astype(numpy.float32)
    gtImage = cv2.cvtColor(gtImage, cv2.COLOR_BGR2GRAY)
    # gtImage = cv2.bitwise_and(gtImage, 0xFFF0)
    gtImage = cv2.cvtColor(gtImage, cv2.COLOR_BGR2RGB)
    # numpy.bitwise_and(numpy.asarray(gtImage), 0xFFF0)
    return gtImage

def NN_cv(image, scale):
    gtImage = gray_12bit_cv(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale

    gtImage = cv2.resize(gtImage, (inCols, inRows), interpolation = cv2.INTER_NEAREST)
    saveFileName = "output/NN_cv.png"
    cv2.imwrite(saveFileName, gtImage) 

def BL_cv(image, scale):
    gtImage = gray_12bit_cv(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale

    gtImage = cv2.resize(gtImage, (inCols, inRows), interpolation = cv2.INTER_LINEAR)
    saveFileName = "output/BL_cv.png"
    cv2.imwrite(saveFileName, gtImage) 

def BC_cv(image, scale):
    gtImage = gray_12bit_cv(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale

    gtImage = cv2.resize(gtImage, (inCols, inRows), interpolation = cv2.INTER_CUBIC)
    saveFileName = "output/BC_cv.png"
    cv2.imwrite(saveFileName, gtImage) 

# ----------------------------------------------------------------------------------------------


# -------------------------------- Image upscalling using Pillow ------------------------------

def NN_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/NN_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.NEAREST)
    gtImage.save(saveFileName)

def BL_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/BL_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.BILINEAR)
    gtImage.save(saveFileName)

def BC_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/BC_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.BICUBIC)
    gtImage.save(saveFileName)

# ----------------------------------------------------------------------------------------------


if __name__ == '__main__':
    scale = 4
    OG_image = "testing/000087.png"

    #---------------------Skimage---------------------------
    time_NN_ski = time.process_time()
    NN_ski(OG_image, scale)
    time_NN_ski = time.process_time() - time_NN_ski

    time_BL_ski = time.process_time()
    BL_ski(OG_image, scale)
    time_BL_ski = time.process_time() - time_BL_ski

    time_BC_ski = time.process_time()
    BC_ski(OG_image, scale)
    time_BC_ski = time.process_time() - time_BC_ski
    #-------------------------------------------------------

    #---------------------Opencv---------------------------
    time_NN_cv = time.process_time()
    NN_cv(OG_image, scale)
    time_NN_cv = time.process_time() - time_NN_cv

    time_BL_cv = time.process_time()
    BL_cv(OG_image, scale)
    time_BL_cv = time.process_time() - time_BL_cv

    time_BC_cv = time.process_time()
    BC_cv(OG_image, scale)
    time_BC_cv = time.process_time() - time_BC_cv
    #-------------------------------------------------------

    #---------------------Pillow---------------------------
    time_NN_PIL = time.process_time()
    NN_pil(OG_image, scale)
    time_NN_PIL = time.process_time() - time_NN_PIL

    time_BL_PIL = time.process_time()
    BL_pil(OG_image, scale)
    time_BL_PIL = time.process_time() - time_BL_PIL

    time_BC_PIL = time.process_time()
    BC_pil(OG_image, scale)
    time_BC_PIL = time.process_time() - time_BC_PIL
    #-------------------------------------------------------


    print("All time is in seconds\n")
    print(tabulate(
            [['Skimage', time_NN_ski, time_BL_ski, time_BC_ski],
            ['OpenCV', time_NN_cv, time_BL_cv, time_BC_cv],
            ['Pillow', time_NN_PIL, time_BL_PIL, time_BC_PIL]],
            headers=['Library', 'NN', 'BL', 'BC'],
            tablefmt='orgtbl'))

