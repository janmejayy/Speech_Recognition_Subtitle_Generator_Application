import glob
import os
import math
#Lrbl2p
# from moviepy.editor import *
import cv2
from pathlib import Path
import ROI
import Features
import Gabor
import traceback
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import datetime
import time
# import subprocess
# import shlex


def Frame(detector, predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath):

    if not os.path.exists(Frame):
        os.mkdir(Frame)
    #obtain the individual words


    video=glob.glob(VideoPath)
    for m in range(0 ,len(video)):
            # print(len(video))
    # for video in glob.glob(VideoPath):  # path of videos
            (filepath, tempfilename) = os.path.split(video[m])
            # print(filepath,tempfilename)
            (video_shotname, extension) = os.path.splitext(tempfilename)
            folder_name = video_shotname
            Path = os.path.join(Frame, folder_name)
            print(Path)
            if not os.path.exists(Path):
                os.mkdir(Path)

            cap = cv2.VideoCapture(video[m])
            fps = cap.get(cv2.CAP_PROP_FPS)
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            i = 0
            count = 0
            a=time.time()
            while (cap.isOpened()):  # cv2.VideoCapture.isOpened()
                i = i + 1
                ret, frame = cap.read()  # cv2.VideoCapture.read()　
                if ret == True:
                    path = Path + '/'
                    picturepath = path + str('%02d' % i) + '.jpg'
                    print(picturepath)
                    # print picturepath

                    cv2.imwrite(picturepath, frame)

                    ROIpath, mouth_centroid_x, mouth_centroid_y, ROI_mouth, widthG, heightG = ROI.rect1(
                                        detector, predictor, i, folder_name, picturepath, MouthPath,GaborPath,SheetPath,FeaturesPath)

                    Gabor_Path = Gabor.Gabor_h( i, ROIpath,
                                               folder_name, GaborPath)
                    Features.Features(mouth_centroid_x, mouth_centroid_y, i, folder_name,
                                      Gabor_Path, SheetPath, FeaturesPath)


                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
