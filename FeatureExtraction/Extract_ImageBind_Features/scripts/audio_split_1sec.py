import shutil
import subprocess
import os
import argparse
import glob

import multiprocessing
from multiprocessing import Pool
from functools import partial
import threading


def JudgeNull(output_root):

    output_list = os.listdir(output_root)
    cnt = 0

    for file_name in output_list:

        audio_path = os.path.join(output_root, file_name)
        audio_sec_files = os.listdir(audio_path)

        nums = len(audio_sec_files)
        print(file_name, nums)
        if nums == 0:
            
            cnt += 1
    print("cnt: ", cnt)



def Extract1Secx(vid_id, dir_path, dst_path):

    print("\nvid_id: ", vid_id)

    input_audio_path = os.path.join(dir_path, vid_id)
    output_audio_path = os.path.join(dst_path, vid_id[:-4])


    if not os.path.exists(output_audio_path):
        os.makedirs(output_audio_path)

    input_audio_path = "\"" + input_audio_path + "\""

    command = 'ffmpeg '
    command += '-i ' + input_audio_path + " "
    command += '-f '
    command += 'segment '
    command += '-segment_time ' + "2 "
    command += '-ar ' + '16000 '
    command += '{0}/%06d.wav'.format(output_audio_path)


    os.system(command)

    return


def MultiProcess(process_count, dir_path, dst_path):

    vid_list = os.listdir(dir_path)
    vid_list.reverse()

    multi_thread = Pool(process_count)
    worker = partial(Extract1Secx, dir_path=dir_path, dst_path=dst_path)
    multi_thread.imap_unordered(worker, vid_list)

    multi_thread.close()
    multi_thread.join()





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_root', type=str, default='/home/data/MUSIC-AVQA/audio_16kHz', help='input root')
    parser.add_argument('--output_root', type=str, default='/home/data/MUSIC-AVQA/audio_16kHz_2sec', help='output root')
    args = parser.parse_args()


    JudgeNull(args.output_root)

    # cpu number
    # cpu_count = multiprocessing.cpu_count()
    # process_count = cpu_count * 2 - 1

    # MultiProcess(process_count, args.input_root, args.output_root)

    print("\n----------------- Finished! -------------------")
