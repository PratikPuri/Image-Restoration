import cv2
import numpy as np
import math

### Test Image Input ###

im = cv2.imread("Messi.jpg", 1)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image_Original', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image_Original',im)

### Gaussian noise and blurring ###

# Noise

row,col = im.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col))

formatted = (gauss * 255 / np.max(gauss)).astype('uint8')
cv2.namedWindow('image_Noise', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image_Noise',formatted)

# Blurring

kx = cv2.getGaussianKernel(5, 0)
kernel = kx*np.matrix.transpose(kx)
kernel1 = kernel.astype(np.float)
im1 = im.astype(np.float)
x=0
for a in range(5):
    for b in range(5):
        x = x + im1[a][b]
kernel1 = kernel1/x
blur = cv2.filter2D(im1,-1,kernel1)
blur = (blur * 255 / np.max(blur)).astype('uint8')

cv2.namedWindow('Blur_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Blur_image',blur)

noisy = cv2.add(blur,formatted)

cv2.namedWindow('Corrupted_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Corrupted_image',noisy)

### Implementation of Wiener filter ###

## Calculation of inverse transfer function in frequency domain

# FFT of blurring transfer function

a = np.fft.fft2(kernel1,(row,col))
c = (np.abs(a))**2

# Calculation of Syy

e = np.fft.fft2(noisy)
e1 = np.matrix.conjugate(e)
syy = (np.multiply(e,e1))/(row*col)

# Training data set for calculation of sx

sx=0
for i in range (1,11):
        
    j = "Training_Images/" + str(i)+ ".jpg"
    b = cv2.imread(j, 1)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    d = np.fft.fft2(b, (row, col))
    d1 = np.matrix.conjugate(d)
    sx = sx + (np.multiply(d,d1))/(row*col)
    
sx = sx/10

# Forming a row x col size matrix for su

su = np.full((row, col), var)

# Inverse transfer function G

s1 = np.divide(su,sx)
den = c+s1
g = np.divide(np.matrix.conjugate(a),den)

# Corrected image

x1 = np.multiply(g,e)
fnl = np.fft.ifft2(x1)
fnl = np.real(fnl)
f1 = (fnl*255 / np.max(fnl)).astype('uint8')

### Gamma correction ###

gamma = 0.3
igamma = 1/gamma
fn2 = (((f1 / 255.0) ** igamma) * 255).astype("uint8")
cv2.namedWindow('Corrected_image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Corrected_image',fn2)

### PSNR and MSE calculation ###

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return (20 * math.log10(255.0 / math.sqrt(mse)),mse)

e = psnr(im,noisy)
print "The PSNR value of the corrupted image is : " + str(e[0])
print "The mean square error for corrupted image is : " + str(e[1])

d = psnr(im,fn2)
print "The PSNR value of the corrected image is :" + str(d[0])
print "The mean square error for corrected image is : " + str(d[1])

cv2.waitKey(0)
cv2.destroyAllWindows()
