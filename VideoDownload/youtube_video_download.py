'''
liguangyao
01/13/2021
download youtube url
'''

# Path to ffmpeg

from urllib import request
import sys
import os.path
import ffmpeg

import skvideo.io
import cv2
import numpy as np
import pafy
import pandas as pd
import soundfile as sf
import subprocess as sp


def download(path_data, name):
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="
    filename_full_video = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename_full_video):
        print(name, " is alreadly download!")
        return

    print( "download the whole video for: ", name)
    command1 = 'youtube-dl --ignore-config '
    command1 += link + " "
    command1 += "-o " + filename_full_video + " "
    command1 += "-f best "
    os.system(command1)

    print ("Finish the video as: " + filename_full_video)


# Set output settings
audio_codec = 'flac'
audio_container = 'flac'
video_codec = 'h264'
video_container = 'mp4'


if __name__ == "__main__":

    list_path = "./youtube_url_list.txt"
    dl_list = []
    with open(list_path, "r") as lp:
        for line in lp:
            line = line.replace("\n", "")
            dl_list.append(line)

    cnt = 0
    for i in range(len(dl_list)):
        ytid = dl_list[i]
        print(ytid)
        download('./downloaded_videos', ytid)
        cnt += 1
    print(cnt)
    print("\n--------------------- finished ----------------------")




