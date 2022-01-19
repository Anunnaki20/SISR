CMPT819 Project - Single Image Super Resolution (SISR)
Author: Andrew Kostiuk
Date: 2017-12-04

sisrPredict.py:  
- Usage:  python sisrPredict.py <input-image-file> -s <2|4> [-d]
- Use existing models (in "models" directory, using 1000 epochs) to generate SISR image
- Input image file name supplied on command line
- Scaling factor (2 or 4) supplied on command line
- Two modes of operation, "downsample" ("-d") and "upsample" (default)
- For downsampling:
  - Performs similar operation as testing phase of sisr.py for specified input image
  - Generates and saves input image (as grayscale), downscaled image, nearest neighbour, bilinear, 
    bicubic, and SISR images in "output" directory
  - Provides MSE, PSNR, SSIM quality measures
  - Example: python3 sisrPredict.py testing/000087.png -s 4 -d
- For upsampling:
  - Generates and saves input image (as grayscale), and upscaled nearest neighbour, bilinear, 
    bicubic, and SISR images in "output" directory
  - No quality measures are reported as ground truth image not available
  - Log file in "output" directory
  - Example: python3 sisrPredict.py testing/000087.png -s 4
