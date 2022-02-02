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
    return gtImage

def BL_ski(image, scale):
    gtImage = gray_12bit_ski(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale
    blImage = skimage.transform.resize(gtImage, (inRows, inCols), order=1)
    saveFileName = "output/BL_ski.png"
    skimage.io.imsave(saveFileName, blImage)
    return gtImage

def BC_ski(image, scale):
    gtImage = gray_12bit_ski(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale
    bcImage = skimage.transform.resize(gtImage, (inRows, inCols), order=3)
    saveFileName = "output/BC_ski.png"
    skimage.io.imsave(saveFileName, bcImage)
    return gtImage

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
    return gtImage

def BL_cv(image, scale):
    gtImage = gray_12bit_cv(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale

    gtImage = cv2.resize(gtImage, (inCols, inRows), interpolation = cv2.INTER_LINEAR)
    saveFileName = "output/BL_cv.png"
    cv2.imwrite(saveFileName, gtImage)
    return gtImage

def BC_cv(image, scale):
    gtImage = gray_12bit_cv(image)
    inRows = gtImage.shape[0] * scale
    inCols = gtImage.shape[1] * scale

    gtImage = cv2.resize(gtImage, (inCols, inRows), interpolation = cv2.INTER_CUBIC)
    saveFileName = "output/BC_cv.png"
    cv2.imwrite(saveFileName, gtImage)
    return gtImage

# ----------------------------------------------------------------------------------------------


# -------------------------------- Image upscalling using Pillow ------------------------------

def NN_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/NN_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.NEAREST)
    gtImage.save(saveFileName)
    return gtImage

def BL_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/BL_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.BILINEAR)
    gtImage.save(saveFileName)
    return gtImage

def BC_pil(image, scale):
    gtImage = Image.open(image).convert("L")
    inRows = gtImage.height * scale
    inCols = gtImage.width * scale
    saveFileName = "output/BC_PIL.png"
    gtImage = gtImage.resize((inRows, inCols), Image.BICUBIC)
    gtImage.save(saveFileName)
    return gtImage

# ----------------------------------------------------------------------------------------------

def DetermineComparisons(image1, image2):
    
    mse = skimage.metrics.mean_squared_error(image1, image2)
    psnr = skimage.metrics.peak_signal_noise_ratio(image1, image2)
    ssim = skimage.metrics.structural_similarity(image1, image2)
    return [mse, psnr, ssim]

def quality_test(original_image_file, scale):

    # Downscale the image with the OG image library
    original_image = skimage.io.imread(original_image_file)
    original_image = skimage.img_as_uint(skimage.color.rgb2gray(skimage.color.rgba2rgb(original_image))) & 0xFFF0
    original_image = skimage.img_as_float(original_image)
    smallImage = skimage.transform.downscale_local_mean(original_image, (scale, scale))

    # Special version for pil
    original_image_pil = Image.open(original_image_file).convert("L")
    smallImage_pil = original_image_pil.resize(( int(original_image_pil.height/scale), int(original_image_pil.width/scale) ), Image.BICUBIC)

    inRows_pil = smallImage_pil.height * scale
    inCols_pil = smallImage_pil.width * scale
    inRows = smallImage.shape[0] * scale
    inCols = smallImage.shape[1] * scale

    NN_ski_image = skimage.transform.resize(smallImage, (inRows, inCols), order=0)
    NN_ski_data = DetermineComparisons(original_image, NN_ski_image)

    NN_cv_image = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_NEAREST)
    NN_cv_data = DetermineComparisons(original_image, NN_cv_image)

    NN_pil_image = smallImage_pil.resize((inRows_pil, inCols_pil), Image.NEAREST)
    NN_pil_data = DetermineComparisons(numpy.array(original_image_pil), numpy.array(NN_pil_image) )  
    print("NearNeighour Data:")
    print(tabulate(
        [['Skimage', NN_ski_data[0], NN_ski_data[1], NN_ski_data[2]],
        ['OpenCV', NN_cv_data[0], NN_cv_data[1], NN_cv_data[2]],
        ['Pillow', NN_pil_data[0], NN_pil_data[1], NN_pil_data[2]]],
        headers=['Library', 'MSE', 'PSNR', 'SSIM'],
        tablefmt='orgtbl'))
    print()
    

    BL_ski_image = skimage.transform.resize(smallImage, (inRows, inCols), order=1)
    BL_ski_data = DetermineComparisons(original_image, BL_ski_image)

    BL_cv_image = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_LINEAR)
    BL_cv_data = DetermineComparisons(original_image, BL_cv_image)

    BL_pil_image = smallImage_pil.resize((inRows_pil, inCols_pil), Image.BILINEAR)
    BL_pil_data = DetermineComparisons(numpy.array(original_image_pil), numpy.array(BL_pil_image))
    print("Bi-Linear Data:")
    print(tabulate(
        [['Skimage', BL_ski_data[0], BL_ski_data[1], BL_ski_data[2]],
        ['OpenCV', BL_cv_data[0], BL_cv_data[1], BL_cv_data[2]],
        ['Pillow', BL_pil_data[0], BL_pil_data[1], BL_pil_data[2]]],
        headers=['Library', 'MSE', 'PSNR', 'SSIM'],
        tablefmt='orgtbl'))
    print()
    

    BC_ski_image = skimage.transform.resize(smallImage, (inRows, inCols), order=3)
    BC_ski_data = DetermineComparisons(original_image, BC_ski_image)

    BC_cv_image = cv2.resize(smallImage, (inCols, inRows), interpolation = cv2.INTER_CUBIC)
    BC_cv_data = DetermineComparisons(original_image, BC_cv_image)

    BC_pil_image = smallImage_pil.resize((inRows_pil, inCols_pil), Image.BICUBIC)
    BC_pil_data = DetermineComparisons(numpy.array(original_image_pil), numpy.array(BC_pil_image))
    print("Bi-Cubic Data:")
    print(tabulate(
        [['Skimage', BC_ski_data[0], BC_ski_data[1], BC_ski_data[2]],
        ['OpenCV', BC_cv_data[0], BC_cv_data[1], BC_cv_data[2]],
        ['Pillow', BC_pil_data[0], BC_pil_data[1], BC_pil_data[2]]],
        headers=['Library', 'MSE', 'PSNR', 'SSIM'],
        tablefmt='orgtbl'))
    print()



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
    print()

    quality_test(OG_image, scale)

