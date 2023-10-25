import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import glob

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip_net.clip.load("ViT-B/32", device=device)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


def clip_feat_extract(img):

	image = preprocess(Image.open(img)).unsqueeze(0).to(device)
	with torch.no_grad():
		image_features = model.encode_image(image)
	return image_features

def VideoLevelPrompt(video_label_list, video_name):

	# pre_img_content = "This image may contain "
	# pre_wav_content = "This sound may contain "

	pre_content = "This image and sound may have "
	video_level_prompt = ''

	for video_label_info in video_label_list:

		video_label_info = video_label_info.split(',')
		video_name_info = video_label_info[0]
		video_labels = video_label_info[1:]

		if video_name == video_name_info:
			
			# print("video_label_info: ", video_name, video_label_info, video_labels)

			if len(video_labels) == 1:
				video_level_prompt  = pre_content + video_labels[0]
			elif len(video_labels) == 2:
				video_level_prompt  = pre_content + video_labels[0] + ' and ' + video_labels[-1]
			else:
				cnt = 0
				video_level_prompt  = pre_content

				for v_label in video_labels:
					if cnt == 0:
						video_level_prompt = video_level_prompt + v_label
					elif cnt < len(video_labels)-1:
						video_level_prompt  = video_level_prompt + ', ' + v_label
					else:
						video_level_prompt = video_level_prompt + ' and ' + v_label
					cnt += 1

	video_level_prompt = video_level_prompt  + "."
	# print("video_level_prompt: ", video_level_prompt)
	video_level_prompt = [video_level_prompt,]

	return video_level_prompt


def ImageBind_feat_extract(dir_audio_path, dir_frame_path, dir_prompt_path, dst_audio_path, dst_frame_path, dst_prompt_path):

	video_label_list = []
	with open(dir_prompt_path, 'r') as dpp:
		for line in dpp:
			video_label_list.append(line.replace("\n", ""))
	# print(video_label_list)

	video_list = os.listdir(dir_frame_path)
	video_idx = 0
	total_nums = len(video_list)

	for video_name in video_list:

		video_idx = video_idx + 1
		print("\n--> ", video_idx, video_name)

		audio_save_file = os.path.join(dst_audio_path, video_name + '.npy')
		frame_save_file = os.path.join(dst_frame_path, video_name + '.npy')
		text_save_file = os.path.join(dst_prompt_path, video_name + '.npy')

		if os.path.exists(audio_save_file):
			print(video_name + '.npy', "is already processed!")
			continue

		frame_list_load = sorted(glob.glob(os.path.join(dir_frame_path, video_name, '*.jpg')))
		audio_list_load = sorted(glob.glob(os.path.join(dir_audio_path, video_name, '*.wav')))
		text_list = VideoLevelPrompt(video_label_list, video_name)


		frame_nums = len(frame_list_load)
		if frame_nums < 60:
			frame_samples = np.round(np.linspace(0, frame_nums-2, 60))
		else:
			frame_samples = np.round(np.linspace(0, 60-1, 60))
		frame_list  = [frame_list_load[int(sample)] for sample in frame_samples]

		audio_nums = len(audio_list_load)
		if audio_nums < 30:
			audio_samples = np.round(np.linspace(0, audio_nums-2, 30))
		else:
			audio_samples = np.round(np.linspace(0, 30-1, 30))
		audio_list  = [audio_list_load[int(sample)] for sample in audio_samples]

		# img_features = torch.zeros(len(img_list), patch_nums, C)


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








if __name__ == "__main__":

	# 158
	# dir_audio_path = '/data/users/guangyao_li/MUSIC-AVQA/audio_16kHz_1sec'
	# dir_frame_path = '/data/users/guangyao_li/MUSIC-AVQA/avqa-frames-1fps'
	# dir_simple_prompt_path = './music_avqa_dataset/video_info/video_weakly_labels.txt'
	# dst_audio_path = '/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_audio_16kHz'
	# dst_frame_path = '/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_frame_1fps'
	# dst_simple_prompt_path = '/data/users/guangyao_li/MUSIC-AVQA/imagebind_feat/imagebind_prompt_simple'

	# 137
	dir_audio_path = '/home/data/MUSIC-AVQA/audio_16kHz_2sec'
	dir_frame_path = '/home/data/MUSIC-AVQA/avqa-frames-1fps'
	dir_simple_prompt_path = './music_avqa_dataset/video_info/video_weakly_labels.txt'

	dst_audio_path = '/home/data/MUSIC-AVQA/imagebind/imagebind_audio_16kHz'
	dst_frame_path = '/home/data/MUSIC-AVQA/imagebind/imagebind_frame_1fps'
	dst_simple_prompt_path = '/home/data/MUSIC-AVQA/imagebind/imagebind_prompt_simple'

	ImageBind_feat_extract(dir_audio_path, dir_frame_path, dir_simple_prompt_path, dst_audio_path, dst_frame_path, dst_simple_prompt_path)


	