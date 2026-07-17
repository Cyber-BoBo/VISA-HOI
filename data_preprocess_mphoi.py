import os
import re
import cv2
import time
import clip

import json
import zarr
import hydra
import torch
import pickle
import base64

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from numcodecs import Blosc
from omegaconf import DictConfig
from torchvision.ops import RoIAlign
from volcenginesdkarkruntime import Ark


from natsort import natsorted
from detr.models import build_model


from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

from constant import *
torch.set_grad_enabled(False)

device = 'cuda'

clip_weight = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/Tools/Image-TextProcess/CLIP/opt/models/ViT-L-14-336px.pt'

# MPHOI-72
class mphoi():
    def __init__(self):
        self.name = 'mphoi'
        self.data_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72'
        # self.caption_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_caption/mphoi_captions_clip_prefix.json'
        self.image_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_raw_datas/MPHOI_rgb'
        self.video_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_raw_datas/MPHOI_videos' 
        self.ground_truth = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_ground_truth_labels.json'
        self.output_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_derived_features'
        self.action = ['sitting', 'approaching', 'retreating', 'lifting', 'placing',
            'pouring', 'drinking' ,'cheering', 'cutting', 'drying', 'working',
            'asking', 'solving']
        
    def find_image_files(self):
        data_dict = {}
        for subject in MPHOI_72_SUBJECT:
            for task in MPHOI_72_TASK:
                for take in MPHOI_72_TAKE:
                    current_video_dir = os.path.join(self.image_dir, subject, task, take)
                    current_video_name = subject + '-' + task + '-' + take
                    data_dict[current_video_name] = os.path.join(self.image_dir, current_video_dir)

        return data_dict

    def img2video(self):
        data_dict = self.find_image_files()
        output_dict = {}
        FPS = 3      
        for key, value in data_dict.items():
            files = os.listdir(value)
            files = natsorted(files)
            OUTPUT_VIDEO = os.path.join(self.video_dir, key + ".mp4")            # 输出视频文件名

            try:
                total_frames = self.frames_to_video(files, value, OUTPUT_VIDEO, fps=FPS)
                output_dict[key] = total_frames
            except Exception as e:
                print(f"ERROR 处理失败：{str(e)}")
        with open(os.path.join(self.video_dir, 'video_frame_count.json'),'w') as f:
            json.dump(output_dict, f)
    
    def frames_to_video(self, all_frames, INPUT_PATH, output_video, fps=3, size =(640, 360)):
        # frames = natsorted(all_frames)
        total_frames = len(all_frames)
        print(f"共发现{total_frames}帧图像，已按序号排序")
        
        # with Image.open(os.path.join(INPUT_PATH, all_frames[0])) as img:
        #     size = img.size  # (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, size)
        
        for i, frame_path in enumerate(all_frames):
            with Image.open(os.path.join(INPUT_PATH, frame_path)) as img:
                resized_img = cv2.resize(np.array(img), size)
                frame = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{total_frames} 帧")
        
        # 释放资源
        out.release()
        cv2.destroyAllWindows()
        print(f"\n 视频生成完成：{output_video}")
        print(f"视频信息：尺寸{size}，帧率{fps}，总帧数{total_frames}")
        return total_frames//fps
    

if __name__ == '__main__':
    # main()
    # CaptionGenerationWithDoubaoAPI()
    dataset = mphoi()
    dataset.img2video()