'''
liguangyao
10/25/2023
guangyaoli@ruc.edu.cn
'''

import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def VideoLevelPrompt(video_label_list, video_name):

    video_level_prompt = 'A photo of a dog.'

    return video_level_prompt


def ImageBind_feat_extract(args, dir_audio_path, dir_viusal_path, dir_text_path, dst_audio_path, dst_visual_path, dst_text_path):

    # 此处为文本
    video_label_list = []
    with open(dir_text_path, 'r') as dpp:
        for line in dpp:
            video_label_list.append(line.replace("\n", ""))
    # print(video_label_list)

    video_list = os.listdir(dir_viusal_path)
    video_idx = 0
    total_nums = len(video_list)

    for video_name in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video_name)

        audio_save_file = os.path.join(dst_audio_path, video_name + '.npy')
        frame_save_file = os.path.join(dst_visual_path, video_name + '.npy')
        text_save_file = os.path.join(dst_text_path, video_name + '.npy')

        if os.path.exists(audio_save_file):
            print(video_name + '.npy', "is already processed!")
            continue

        frame_list_load = sorted(glob.glob(os.path.join(dir_viusal_path, video_name, '*.jpg')))
        audio_list_load = sorted(glob.glob(os.path.join(dir_audio_path, video_name, '*.wav')))
        text_list = VideoLevelPrompt(video_label_list, video_name) # 例如：A photo of a dog. 保证文本是陈述语句即可，可自行设计

        # 为了保证模型训练可以批处理，故需要保证每个数据样本后的长度一致。
        # 然而由于不同的视频长度不一，采样出的帧数不一致，故此处对每个视频进行均匀采样。
        frame_nums = len(frame_list_load)
        if frame_nums < args.frame_nums:
            frame_samples = np.round(np.linspace(0, frame_nums-2, args.frame_nums))
        else:
            frame_samples = np.round(np.linspace(0, args.frame_nums-1, args.frame_nums))
        frame_list  = [frame_list_load[int(sample)] for sample in frame_samples]

        audio_nums = len(audio_list_load)
        if audio_nums < args.audio_nums:
            audio_samples = np.round(np.linspace(0, audio_nums-2, args.audio_nums))
        else:
            audio_samples = np.round(np.linspace(0, args.audio_nums-1, args.audio_nums))
        audio_list  = [audio_list_load[int(sample)] for sample in audio_samples]

        # Load data
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            ModalityType.VISION: data.load_and_transform_vision_data(frame_list, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)


        text_feat = embeddings['text']
        audio_feat = embeddings['audio']
        visual_feat = embeddings['vision']

        # print("\nimagebind text: ", text_feat.shape)
        # print("imagebind audio: ", audio_feat.shape)
        # print("imagebind visual: ", visual_feat.shape)

        text_feat = text_feat.float().cpu().numpy()
        np.save(text_save_file, text_feat)

        audio_feat = audio_feat.float().cpu().numpy()
        np.save(audio_save_file, audio_feat)

        visual_feat = visual_feat.float().cpu().numpy()
        np.save(frame_save_file, visual_feat)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx)
        print("T-A-V Feat shape: ", text_feat.shape, audio_feat.shape, visual_feat.shape)


def ImageBind_visaul_feat_extract(args, dir_viusal_path, dst_visual_path):

    video_list = os.listdir(dir_viusal_path)
    video_idx = 0
    total_nums = len(video_list)

    for video_name in video_list:

        video_idx = video_idx + 1
        print("\n--> ", video_idx, video_name)

        frame_save_file = os.path.join(dst_visual_path, video_name + '.npy')

        if os.path.exists(frame_save_file):
            print(video_name + '.npy', "is already processed!")
            continue

        frame_list_load = sorted(glob.glob(os.path.join(dir_viusal_path, video_name, '*.jpg')))

        frame_nums = len(frame_list_load)
        if frame_nums < args.frame_nums:
            frame_samples = np.round(np.linspace(0, frame_nums-2, args.frame_nums))
        else:
            frame_samples = np.round(np.linspace(0, args.frame_nums-1, args.frame_nums))
        frame_list  = [frame_list_load[int(sample)] for sample in frame_samples]

        # Load data
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(frame_list, device),}

        with torch.no_grad():
            embeddings = model(inputs)

        visual_feat = embeddings['vision']
        # print("imagebind visual: ", visual_feat.shape)

        visual_feat = visual_feat.float().cpu().numpy()
        np.save(frame_save_file, visual_feat)

        print("Process: ", video_idx, " / ", total_nums, " ----- video id: ", video_idx)
        print("V Feat shape: ", visual_feat.shape)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dir_audio_path", type=str, default='data/users/guangyao_li/MUSIC-AVQA/audio_16kHz_2sec',
                        help='audio file path')
    parser.add_argument("--dir_visual_path", type=str, default='/data/users/guangyao_li/MUSIC-AVQA/avqa-frames-1fps',
                        help='visual frames path')
    parser.add_argument("--dir_text_path", type=str, default='../../dataset/split_que_id/music_avqa.json',
                        help='text file path')

    parser.add_argument("--dst_audio_path", type=str, default='/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_audio_16kHz',
                        help='audio feature path')
    parser.add_argument("--dst_visual_path", type=str, default='/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_frame_1fps',
                        help='visual frames feature path')
    parser.add_argument("--dst_text_path", type=str, default='/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_text',
                        help='text feature path')

    parser.add_argument("--frame_nums", type=int, default=60,
                        help='frame sample numbers')
    parser.add_argument("--audio_nums", type=int, default=60,
                        help='audio clip sample numbers')
    # parser.add_argument("--gpu", dest='gpu', type=str, default='0',
    #                     help='Set CUDA_VISIBLE_DEVICES environment variable, optional')

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)

    # 同时提取audio, vsiual 和text的特征
    ImageBind_feat_extract(args, args.dir_audio_path, args.dir_visual_path, args.dir_text_path, 
                           args.dst_audio_path, args.dst_visual_path, args.dst_text_path)

    # 只提取一个模态的特征，如visual
    ImageBind_visual_feat_extract(args, dir_viusal_path, dst_visual_path)


    