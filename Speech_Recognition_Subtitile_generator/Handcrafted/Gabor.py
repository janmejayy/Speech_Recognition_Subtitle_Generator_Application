
import cv2
import numpy as np
import pylab as pl
from PIL import Image
from PIL import ImageFilter
import os
import glob
import Features

def Gabor_h(i,ROIpath,shotname,GaborPath):

    #cur_dir2 = 'D:/codepython3/Gabor/' # path to store Gabor features

    if not os.path.exists(GaborPath):
        os.mkdir(os.path.join(GaborPath))
    Gaborpath = os.path.join(GaborPath, shotname)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)

    img = cv2.imread(ROIpath, 1)  # Loading color picture
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Change color picture into gray picture
    imgGray_f = np.array(imgGray, dtype=np.float32)  # Change data type of picture
    imgGray_f /= 255.

    # Parameters of horizontal filter
    # "Hkernel_size": ('Hkernel_size', 5, 20, 1),
    # "Hwavelength": ('Hwavelength', 5, 20, 1),
    # "Hsig": ('Hsig', 3, 7, 1),
    # "Hgamma": ('Hgamma', 0.3, 0.7, 0.1)

    orientationH = 90  # orientation of normal direction
    wavelengthH = 15
    kernel_sizeH = 16
    sigH = 4
    gmH = 0.3

    ps = 0.0
    thH = orientationH * np.pi / 180
    kernelH = cv2.getGaborKernel((kernel_sizeH, kernel_sizeH), sigH, thH, wavelengthH, gmH, ps)
    destH = cv2.filter2D(imgGray_f, cv2.CV_32F, kernelH)  # CV_32F
    Gaborpath = Gaborpath + '/'   # 保存路径
    # print(path)
    if not os.path.exists(Gaborpath):
        os.mkdir(Gaborpath)
    Gabor_Path = Gaborpath  + str('%02d' % i) + '.jpg'
    cv2.imwrite(Gabor_Path, np.power(destH, 2))
    return Gabor_Path

    #return i,shotname,Gabor_Path