
import cv2
import os
import pydicom
import numpy as np

inputdir = 'Harvard_images/Subject (1)/98.12.2/'
outdir = 'test_images/'
#os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]

for f in test_list[:10]:   # remove "[:10]" to convert all images 
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array.astype(np.uint16) # get image array
    scaled_img = (np.maximum(img,0) / img.max()) * 255.0
    cv2.imwrite(outdir + f.replace('.dcm','.png'), np.uint8(scaled_img)) # write png image