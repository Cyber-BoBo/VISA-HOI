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

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from dtw import dtw
from fastdtw import fastdtw
from tslearn.metrics import dtw_path


from PIL import Image
from tqdm import tqdm
from numcodecs import Blosc
from omegaconf import DictConfig
from torchvision.ops import RoIAlign
from volcenginesdkarkruntime import Ark

from scipy.spatial.distance import euclidean, cdist

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
            OUTPUT_VIDEO = os.path.join(self.video_dir, key + ".mp4")            # 输出视频文件名

            try:
                total_frames = self.frames_to_video(value, OUTPUT_VIDEO, fps=FPS)
                output_dict[key] = total_frames
            except Exception as e:
                print(f"ERROR 处理失败：{str(e)}")
        with open(os.path.join(self.video_dir, 'video_frame_count.json'),'w') as f:
            json.dump(output_dict, f)
    
    def frames_to_video(self, all_frames, output_video, fps=3, size =None):
        frames = natsorted(all_frames)
        total_frames = len(frames)
        print(f"共发现{total_frames}帧图像，已按序号排序")
        
        with Image.open(frames[0]) as img:
            size = img.size  # (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, size)
        
        for i, frame_path in enumerate(frames):
            with Image.open(frame_path) as img:
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{total_frames} 帧")
        
        # 释放资源
        out.release()
        cv2.destroyAllWindows()
        print(f"\n 视频生成完成：{output_video}")
        print(f"视频信息：尺寸{size}，帧率{fps}，总帧数{total_frames}")
        return total_frames//fps

class cad120():
    def __init__(self):
        self.name = 'cad120'
        self.data_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/CAD-120'
        # self.caption_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_caption/mphoi_captions_clip_prefix.json'
        self.dictionary = os.path.join(self.data_path, 'dictionaries')
        self.features = os.path.join(self.data_path, 'features')
        self.output_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/CAD-120/features'
        self.action = ['reaching', 'moving', 'pouring', 'eating', 'drinking',
                        'opening', 'placing' ,'closing', 'null', 'cleaning']
        self.video_id = ['0510141923','0510142045','1204175902','0510144057','0510144139','0510141947','0510144324','0510144350',
                         '1204175103','1204175622','1204174740','1204175316','1204174554','0510143446','1204175712','0510144215',
                         '0510142800','0510142419','1204174844','0510144450','1204180344','1204173846','0510143426','1204180612',
                         '0510142336','1204173536','1204174024','1204174314','0510143618','1204180515','1204175451','1130144557',
                         '0510173203','0510171810','1130151154','1130150747','0510172425','0510172745','0510173714','1130144713',
                         '1130144814','1130145737','0510173634','1130150135','1130151710','0510172333','0510173051','0510172851',
                         '1130151025','0510172015','1130151500','0510172725','0510173217','0204141211','0510172557','1130145835',
                         '0204140840','1130144242','0510173506','0510172049','0204141007','1130151121','1204142858','1204144410',
                         '1204150645','1204151136','0510181236','0510180532','1204144736','1204144120','0510175921','0510180218',
                         '0510182137','1204142616','0510175411','1204145902','1204150828','0510175855','1204143959','0510175829',
                         '0510182019','0510180342','1204142055','1204142227','0510182057','0510181415','0510175431','1204145630',
                         '1204145234','0510181310','0510175554','1204142500','1204145527','0504235908','0511141007','0126142037',
                         '0129114153','0511140553','0504233253','0129112342','0129112630','0126141638','0126143115','0126141850',
                         '0126142253','0129112226','0504232829','0511141231','0511141338','0504235245','0511140450','0504235647',
                         '0126143251','0505002750','0504233320','0511140410','0129114356','0129111131','0505003237','0505002942',
                         '0129112522','0126143431','0129114054','0129112015' ]

class bimanualactions():
    def __init__(self):
        self.name = 'bimanual'
        self.data_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions'
        # self.caption_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_caption/mphoi_captions_clip_prefix.json'
        self.image_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_raw_datas'
        self.video_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_videos'
        self.ground_truth = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_ground_truth_labels.json'
        self.output_dir = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_derived_features'
        self.action = ["idle", "approach", "retreat", "lift", "place",
                        "hold", "pour", "cut", "hammer", "saw", 
                        "stir", "screw", "drink", "wipe"]
        
    def find_image_files(self):
        data_dict = {}
        for subject in BIMACS_SUBJECT:
            for task in BIMACS_TASK:
                for take in BIMACS_TAKE:
                    current_video_dir = os.path.join(self.image_dir, subject, task, take, 'rgb')
                    current_video_name = subject + '-' + task + '-' + take
                    data_dict[current_video_name] = os.path.join(self.image_dir, current_video_dir)

        return data_dict
    
    def img2video(self):
        data_dict = self.find_image_files()
        output_dict = {}
        FPS = 3      
        for key, value in data_dict.items():
            INPUT_RGB_FOLDER = value  # 例如 "D:/video_project/rgb"
            OUTPUT_VIDEO = os.path.join(self.video_dir, key + ".mp4")            # 输出视频文件名

            try:
                total_frames = self.chunked_frames_to_video(INPUT_RGB_FOLDER, OUTPUT_VIDEO, fps=FPS)
                output_dict[key] = total_frames
            except Exception as e:
                print(f"ERROR 处理失败：{str(e)}")
        with open(os.path.join(self.video_dir, 'video_frame_count.json'),'w') as f:
            json.dump(output_dict, f)

    def chunked_frames_to_video(self, rgb_folder, output_video, fps=3, size=None):
        
        chunk_folders = [f for f in os.listdir(rgb_folder) 
                        if f.startswith('chunk_') and os.path.isdir(os.path.join(rgb_folder, f))]
        if not chunk_folders:
            raise ValueError("未找到任何chunk子文件夹，请检查rgb目录")
        chunk_folders.sort(key=lambda x: int(x.split('_')[1]))
        
        
        all_frames = []
        for chunk in chunk_folders:
            chunk_path = os.path.join(rgb_folder, chunk)
            frame_files = [f for f in os.listdir(chunk_path) 
                        if f.startswith('frame_') and f.lower().endswith('.png')]
            
            for frame_file in frame_files:
                frame_path = os.path.join(chunk_path, frame_file)
                all_frames.append(frame_path)
        
        if not all_frames:
            raise ValueError("未找到任何frame图像，请检查文件命名格式")
        
        frames = natsorted(all_frames)
        total_frames = len(frames)
        print(f"共发现{total_frames}帧图像，已按序号排序")
        
        with Image.open(frames[0]) as img:
            size = img.size  # (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, size)
        
        for i, frame_path in enumerate(frames):
            with Image.open(frame_path) as img:
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{total_frames} 帧")
        
        # 释放资源
        out.release()
        cv2.destroyAllWindows()
        print(f"\n 视频生成完成：{output_video}")
        print(f"视频信息：尺寸{size}，帧率{fps}，总帧数{total_frames}")
        return total_frames//fps


image_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(800),      # 将图像的较小边缩放到800像素，同时保持长宽比
    T.ToTensor(),       # 将图像从PIL格式或numpy数组转换为PyTorch张量，并且将图像的像素值从0-255缩放到0-1之间
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # 使用ImageNet数据集上的均值和标准差对图像进行归一化
    # 均值为[0.485, 0.456, 0.406]，标准差为[0.229, 0.224, 0.225]
])

def pad_dim_to_torch(arr: torch.Tensor, dim: int, target_length: int) -> torch.Tensor:
    cuurent_length = arr.size(dim)
    pad_size = target_length - cuurent_length

    n_dims = arr.ndim
    if pad_size < 0:
        slices = [slice(None)] * n_dims
        slices[dim] = slice(0, target_length)
        return arr[slices]
    else:
        pad = [0] * (2 * n_dims)
        pad_position = (n_dims - 1 - dim) * 2 + 1
        pad[pad_position] = pad_size

        return F.pad(arr, tuple(pad), value=0)

def sort_by_scores(result, de_hs, en_hs, max_num=100):
    boxes = result['boxes'].cpu()
    labels = result['labels'].cpu()
    scores = result['scores'].cpu()
    de_hs = de_hs.cpu()
    data_info = []
    for i in range(boxes.shape[0]):
        data_info.append([boxes[i], labels[i], scores[i], de_hs[:,:,i,:]])
    data_info.sort(key=lambda x: x[2], reverse=True)
    data_info = data_info[:max_num]

    for i in range(len(data_info)):
        if i == 0:
            boxes_out = data_info[i][0].unsqueeze(0)
            labels_out = data_info[i][1].unsqueeze(0)
            scores_out = data_info[i][2].unsqueeze(0)
            de_hs_out = data_info[i][3].unsqueeze(0)
        else:
            boxes_out = torch.cat((boxes_out, data_info[i][0].unsqueeze(0)), dim=0)
            labels_out = torch.cat((labels_out, data_info[i][1].unsqueeze(0)), dim=0)
            scores_out = torch.cat((scores_out, data_info[i][2].unsqueeze(0)), dim=0)
            de_hs_out = torch.cat((de_hs_out, data_info[i][3].unsqueeze(0)), dim=0)

    de_hs_out = de_hs_out.squeeze(2).permute(1,0,2).to('cpu')
    en_hs_out = en_hs.squeeze(0).to('cpu')
    boxes_out = boxes_out.to('cpu')
    labels_out = labels_out.squeeze(1).to('cpu')
    scores_out = scores_out.squeeze(1).to('cpu')

    return boxes_out, labels_out, scores_out, de_hs_out, en_hs_out

# 用于输出边界框的后处理
def box_cxcywh_to_xyxy(x):
    "将边界框从中心坐标和宽高格式转换为左上角-右下角格式"
    x_c, y_c, w, h = x.unbind(2)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    "根据图像大小对边界框进行缩放"
    img_w, img_h = size
    
    b = box_cxcywh_to_xyxy(out_bbox).squeeze(0)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).unsqueeze(1).to('cuda:0')

    b = b * scale
    return b
# 定义一个函数，

def post_process(outputs, im_size):
    """ Perform the computation
    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                        For evaluation, this must be the original image size (before any data augmentation)
                        For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    # convert to [x0, y0, x1, y1] format
    # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates

    boxes = rescale_bboxes(out_bbox, im_size).permute(1,0)
    labels = labels.permute(1,0)
    scores = scores.permute(1,0)
    # print(boxes.shape)

    results = {'scores': scores, 'labels': labels, 'boxes': boxes}
    # print(results[5])
    return results

def detect(im, model, transform):
    # 对输入图像进行均值-标准差归一化（批量大小：1）
    t0 = time.time()
    im_size = torch.tensor(im.shape[:-1])
    img = transform(im).unsqueeze(0)
    # 使用定义的transform对输入图像进行归一化处理，并增加一个batch维度
 
    # 演示模型默认仅支持宽高比在0.5到2之间的图像, 如果你想使用宽高比在此范围之外的图像,需要重新调整图像的大小，以确保最大尺寸不超过1333像素，效果最佳
    # assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, '演示模型仅支持每边最多1600像素的图像'
    # 确保输入图像的长宽不超过1600像素
 
    img = img.to('cuda:0')
    model = model.to('cuda:0')
    outputs, de_hs, en_hs = model(img)
    result = post_process(outputs, im_size)

    return result, de_hs, en_hs







# Image Caption and Text Encoding
def generate_caption(image_path, model, processor):
    
    # processing
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    
    # caption generation
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def clip_encode_text(text, model, processor):
    
    # processing
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**inputs)

    return text_features.detach()

def image_caption_text_encoding_4_one_image(image_path, clip_model, blip_model, clip_processor, blip_processor):

    caption = generate_caption(image_path, blip_model, blip_processor)
    caption_features = clip_encode_text(caption, clip_model, clip_processor)
    
    return caption_features, caption

def CaptionFeatureEncode(data_set = mphoi(), caption_path = None, generate_caption = False):    
    if data_set.name in {'mphoi','bimanual'}:
        data_dict = data_set.find_image_files()

    clip_model = CLIPModel.from_pretrained("./Image-TextProcess/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("./Image-TextProcess/clip-vit-large-patch14")

    if generate_caption and caption_path is None:
        blip_processor = BlipProcessor.from_pretrained("./Image-TextProcess/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("./Image-TextProcess/blip-image-captioning-large").to(device)

        for key, value in data_dict.items():
            print(key)
            # caption_dict[key] = []
            caption_feature_dict[key] = []
            files = os.listdir(value)
            files.sort(key=lambda x: int(x.split('.')[0][-5:]))
            for file in tqdm(files):
                caption_feature, caption=image_caption_text_encoding_4_one_image(os.path.join(value, file), clip_model, blip_model, clip_processor, blip_processor)
                caption_dict[key].append(caption)
                caption_feature_dict[key].append(caption_feature)
            caption_feature_dict[key] = torch.stack(caption_feature_dict[key],dim=0).cpu()

    elif caption_path is not None:
        caption_dict = {}
        caption_feature_dict = {}
        with open(caption_path, 'r') as f:
            caption_dict = json.load(f)
            for key, captions in caption_dict.items():
                caption_feature_dict[key] = []
                print(key)
                for caption in tqdm(captions):
                    caption_feature = clip_encode_text(caption, clip_model, clip_processor)
                    caption_feature_dict[key].append(caption_feature)
                caption_feature_dict[key] = torch.stack(caption_feature_dict[key],dim=0).cpu()

    # save zarr
    if generate_caption:
        caption_feature_save_name = 'caption_features_clip_prefix.zarr'
        # caption_feature_save_name = 'caption_features_doubao.zarr'
    else:
        caption_feature_save_name = 'caption_realtime.zarr'
        # caption_feature_save_name = 'caption_features_doubao.zarr'
    caption_feature_save_path = os.path.join(data_set.output_dir, caption_feature_save_name)
    caption_feature_store = zarr.DirectoryStore(caption_feature_save_path)
    caption_feature_root = zarr.group(store = caption_feature_store)
    for key, value in caption_feature_dict.items():
        caption_feature_root.create_dataset(
                name = key,
                data = value,
                chunks = (1, 768),
                dtype = 'float32'
                )
    
    # save json
    if generate_caption:
        caption_save_name = 'caption_generated.json'
        caption_save_path = os.path.join(data_set.output_dir, caption_save_name)
        with open(caption_save_path, 'w') as f:
            json.dump(caption_dict, f)


def ActionFeatureEncode(data_set = mphoi()):

    model = CLIPModel.from_pretrained("./Image-TextProcess/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("./Image-TextProcess/clip-vit-large-patch14")

    save_name = 'action_features.zarr'
    action_feature = []
    if data_set.name == 'mphoi':
        for action in tqdm(data_set.action):
            action_feature.append(clip_encode_text('Human1 is ' + action, model, processor))
        for action in tqdm(data_set.action):
            action_feature.append(clip_encode_text('Human2 is ' + action, model, processor))
    elif data_set.name == 'bimanual':
        for action in tqdm(data_set.action):
            action_feature.append(clip_encode_text('Left hand is ' + action, model, processor))
        for action in tqdm(data_set.action):
            action_feature.append(clip_encode_text('Right hand is ' + action, model, processor))
    else:
        for action in tqdm(data_set.action):
            action_feature.append(clip_encode_text('Human is ' + action, model, processor))
    action_feature = torch.stack(action_feature, dim=0).squeeze(1).to('cpu').numpy()

    # save zarr
    save_path = os.path.join(data_set.output_dir, save_name)
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store = store)
    root.create_dataset(
            name = 'action_feature',
            data = action_feature,
            chunks = (1, 768),
            dtype = 'float32'
            )

def LabelTransform(data_set = cad120()):
    data_path = os.path.join(data_set.features,'preprocessed','cad120data.pickle')
    features_path = os.path.join(data_set.features,'faster_rcnn','features.zarr')
    output_path = os.path.join(data_set.data_path,'cad120_labels.json')
    
    max_len = 0
    with open(data_path, mode='rb') as f:
        data = pickle.load(f)
    root = zarr.open(features_path, mode='r')
    
    label_dict = {}
    label_dict['recognition_subactivity'] = {}
    label_dict['prediction_subactivity'] = {}
    label_dict['recognition_affordance'] = {}
    label_dict['prediction_affordance'] = {}
    for video_id, video_data in data.items():
        if video_id in data_set.video_id:
            x_human = root[video_id + '/skeleton'][:]
            x_object = root[video_id + '/objects'][:]
            # max_len = max(max_len, x_human.shape[0])
            label_dict['recognition_subactivity'][video_id] = np.full([x_human.shape[0]], fill_value=-1, dtype=np.int64)
            label_dict['prediction_subactivity'][video_id] = np.full([x_human.shape[0]], fill_value=-1, dtype=np.int64)
            label_dict['recognition_affordance'][video_id] = np.full([x_object.shape[0], x_object.shape[1]], fill_value=-1, dtype=np.int64)
            label_dict['prediction_affordance'][video_id] = np.full([x_object.shape[0], x_object.shape[1]], fill_value=-1, dtype=np.int64)
            for i, video_segment in enumerate(video_data):
                start_frame, end_frame = video_segment.start_frame, video_segment.end_frame
                if start_frame is None or end_frame is None:
                    continue
                start_frame -= 1
                end_frame -= 1
                subactivity = video_segment.subactivity - 1
                next_subactivity = video_segment.next_subactivity
                next_subactivity = next_subactivity - 1 if next_subactivity is not None else -1
                label_dict['recognition_subactivity'][video_id][start_frame:end_frame + 1] = subactivity
                label_dict['prediction_subactivity'][video_id][start_frame:end_frame + 1] = next_subactivity
                affordances = video_segment.object_affordance
                for object_id, object_affordance in affordances.items():
                    label_dict['recognition_affordance'][video_id][start_frame:end_frame + 1, object_id - 1] = object_affordance - 1
                next_affordances = video_segment.next_object_affordance
                for object_id, object_affordance in next_affordances.items():
                    label_dict['prediction_affordance'][video_id][start_frame:end_frame + 1, object_id - 1] = object_affordance - 1
            
            label_dict['recognition_subactivity'][video_id] = label_dict['recognition_subactivity'][video_id].tolist()
            label_dict['prediction_subactivity'][video_id] = label_dict['prediction_subactivity'][video_id].tolist()
            label_dict['recognition_affordance'][video_id] = label_dict['recognition_affordance'][video_id].tolist()
            label_dict['prediction_affordance'][video_id] = label_dict['prediction_affordance'][video_id].tolist()

    with open(output_path, 'w') as f:
        json.dump(label_dict, f)


def Label2Caption(data_set = mphoi()):

    caption = {}
    if data_set.name == 'mphoi':
        label_path = os.path.join(data_set.data_path,'mphoi_ground_truth_labels.json')
        output_path = os.path.join(data_set.data_path,'mphoi_ground_truth_captions.json')
        with open(label_path, 'r') as f:
            labels = json.load(f)
        for key, value in labels.items():
            caption[key] = []
            for i in range(len(value['Human1'])):
                # if value['Human1'][i]==value['Human2'][i]:
                #     caption[key].append('Human1 and Human2 is ' + data_set.action[value['Human1'][i]])
                caption[key].append('Human1 is '+data_set.action[value['Human1'][i]]+' and Human2 is '+data_set.action[value['Human2'][i]])
                # else:
                    # caption[key].append('Human1 is '+data_set.action[value['Human1'][i]]+' and Human2 is '+data_set.action[value['Human2'][i]])

    elif data_set.name == 'cad120':
        label_path = os.path.join(data_set.data_path,'cad120_ground_truth_labels.json')
        output_path = os.path.join(data_set.data_path,'cad120_ground_truth_captions.json')
        with open(label_path, mode='rb') as f:
            data = json.load(f)
        for key, value in data['recognition_subactivity'].items():
            caption[key] = []
            for i in range(len(value)):
                if value[i] == -1 or value[i] == 8:
                    caption[key].append('null')
                else:
                    caption[key].append('Human is ' + data_set.action[value[i]])
    
    elif data_set.name == 'bimanual':
        label_path = os.path.join(data_set.data_path,'bimacs_ground_truth_labels.json')
        output_path = os.path.join(data_set.data_path,'bimacs_ground_truth_captions.json')
        with open(label_path, mode='rb') as f:
            labels = json.load(f)
        for key, value in labels.items():
            caption[key] = []
            for i in range(len(value['left_hand'])):
                # if value['left_hand'][i]==value['right_hand'][i]:
                    # caption[key].append('Left and right hand is ' + data_set.action[value['left_hand'][i]])
                # else:
                caption[key].append('Left hand is '+data_set.action[value['left_hand'][i]]+' and right hand is '+data_set.action[value['right_hand'][i]])

        
    with open(output_path, 'w') as f:
        json.dump(caption, f)

def Image2VisualFeature(data_set = mphoi()):
    with open(data_set.ground_truth, 'r') as f:
        labels = json.load(f)
    # data_dict = data_set.find_image_files()
    label_dict = [[]for i in range(len(MPHOI_72_ACTION))]
    for key, value in labels.items():
        for i in range(len(value['Human1'])):
            label_dict[value['Human1'][i]].append(key+'-frame_'+str(i).zfill(5))
            label_dict[value['Human2'][i]].append(key+'-frame_'+str(i).zfill(5))
    with open(os.path.join(data_set.output_dir, 'mphoi_label2frame.json'), 'w') as f2:
        json.dump(label_dict, f2)


def ROIAlignProcess(data_set = mphoi()):
    # data_dict = data_set.find_image_files()
    visual_features = zarr.open(os.path.join(data_set.output_dir, 'visual_features_DETR.zarr'), mode='r')
    ROI_Aligner = RoIAlign(output_size=(7,7), spatial_scale=1/16, sampling_ratio=2)
    for key in visual_features.keys():
        print(0)


def ImageActionSimilarity(data_set = mphoi()):
    data_dict = data_set.find_image_files()

    model, preprocess = clip.load(clip_weight, device=device)
    text = clip.tokenize(MPHOI_72_ACTION).to(device)

    probs = {}
    for key, value in data_dict.items():
        print(key)
        # caption_dict[key] = []
        probs[key] = []
        files = os.listdir(value)
        files.sort(key=lambda x: int(x.split('.')[0][-5:]))
        for file in tqdm(files):
            image = preprocess(Image.open(os.path.join(value, file))).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                # text_features = model.encode_text(text)
                logits_per_image, logits_per_text = model(image,text)
                probility = logits_per_image.softmax(dim=-1).cpu().tolist()
            probs[key].append(probility)
        # probs[key] = torch.stack(probs[key],dim=0).cpu().numpy()
    
    # save json
    similarity_save_name = 'similarity.json'
    similarity_save_path = os.path.join(data_set.output_dir, similarity_save_name)
    with open(similarity_save_path, 'w') as f:
        json.dump(probs, f)

def GetVisualFeatureDETR(data_set = mphoi(), args = None):
    data_dict = data_set.find_image_files()
    # data save config
    total_length = 30

    # load DETR
    detr_model, _, post_processer = build_model(args.parameters)
    state_dict = torch.load('./detr/weight/detr-r101-dc5-a2e86def.pth', map_location=torch.device(device))
    detr_model.load_state_dict(state_dict['model'])
    detr_model.eval()

    # data compressor
    compressor_features = Blosc(cname = 'lz4', clevel = 3, shuffle = Blosc.BITSHUFFLE)
    
    t_start = time.time()
    en_hs = []
    visual_feature_dict = {}
    #==================================================================================================
    
    # save zarr
    visual_feature_save_name = 'visual_features_DETR.zarr'
    visual_feature_save_path = os.path.join(data_set.output_dir, visual_feature_save_name)
    visual_feature_store = zarr.DirectoryStore(visual_feature_save_path)
    visual_feature_root = zarr.group(store = visual_feature_store)

    i = 0
    for key, value in data_dict.items():
        save_data = []
        files = os.listdir(value)
        files.sort(key=lambda x: int(x.split('.')[0][-5:]))
        for file in tqdm(files):
            
            raw_image = cv2.imread(os.path.join(value, file))
            results, _, en_hs = detect(raw_image, detr_model, image_transform)
            if i ==0:
                height, width  = en_hs.shape[-2:]
            save_data.append(en_hs)
        save_data = torch.stack(save_data,dim=1).cpu().squeeze(0).numpy()
        visual_feature_root.create_dataset(
                name = key,
                data = save_data,
                chunks = (1, 256, height, width),
                dtype = 'float32'
                )
        del save_data
        i= i+1
        print(f'processed {i}-th videos')

    print(f'processing finished in {time.time()- t_start} s.')

def Caption_downsampling(caption_path, output_path, down_sampling =3):
    with open(caption_path, mode='rb') as f:
        captions = json.load(f)
    for key, value in captions.items():
        captions[key] = captions[key][down_sampling-1::down_sampling]
    with open(output_path, 'w') as f:
        json.dump(captions, f)

def encode_video_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def CaptionGenerationWithDoubaoAPI(data_set = bimanualactions()):
    
    video_frame_dict = os.path.join(data_set.video_dir, 'video_frame_count.json')
    with open(video_frame_dict, 'r') as f:
        video_frames = json.load(f)

    start_epsoid = 0
    chat_epsoid = 100
    output_dict = {}

    # 初始化一个Client对象，从环境变量中获取API Key
    # client = Ark(api_key=os.getenv('ARK_API_KEY'))
    client = Ark(api_key='542ce34e-7b77-4212-bd0b-fd996987f01f')
    print('API client 创建成功，准备发送请求...')
    epsoid_count = 0
    for video_id, frames_num in video_frames.items():
        if epsoid_count < start_epsoid:
            epsoid_count += 1
            continue
        elif epsoid_count >=start_epsoid + chat_epsoid:
            break
        print(f'开始处理第{epsoid_count}个视频: {video_id}, 共{frames_num}帧')
        video_path = os.path.join(data_set.video_dir, video_id + '.mp4')
        base64_video = encode_video_base64(video_path)
        
        # 调用 Ark 客户端的 chat.completions.create 方法创建聊天补全请求
        response = client.chat.completions.create(
            # 替换 <MODEL> 为模型的Model ID
            model="doubao-seed-1-6-250615",
            messages=[
                {
                    # 消息角色为用户
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            # 第一张图片链接及细节设置为 high
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}",
                                "fps": 1, # 每秒截取1帧画面，用于视频理解
                                "detail": 'low',
                            }
                        },
                        # 文本类型的消息内容，询问图片里有什么
                        # {"type": "text", "text": f"This is a video with {frames_num} frames, simply describe each frame in video in one sentence strictly follow the format as: Left hand is [doing] and right hand is [doing]. where [doing] is only a single verb in [idle, approach, retreat, lift, place, hold, pour, cut, hammer, saw, stir, screw, drink, wipe]."},
                        {"type": "text", "text": f"This is a video with {frames_num} frames, simply describe each frame in video in one sentence strictly follow the format as: Human1 is [doing] and Human2 is [doing]. where [doing] is only a single verb in [sitting, approaching, retreating, lifting, placing, pouring, drinking, cheering, cutting, drying, working, asking, solving]."},
                    ],
                }
            ],
        )
        epsoid_count += 1

        output_dict[video_id] = response.choices[0].message.content
        print(f'第{epsoid_count}个视频: {video_id}, 处理结束')
        # print(output_dict[video_id])

    with open(os.path.join(data_set.data_path, 'mphoi_caption_doubao_unprocessed.json'),'w') as f:
        try:
            json.dump(output_dict, f)
        except TypeError as e:
            print('保存json失败,暂存为txt')
    with open(os.path.join(data_set.data_path, f'mphoi_caption_doubao_unprocessed_{start_epsoid + chat_epsoid}.txt'),'w', encoding='utf-8') as f:
        for key,value in output_dict.items():
            f.write(key + '\n' + str(value) + '\n')


def constrained_dtw_path(distance_matrix):
    """
    基于约束动态规划的矩阵路径搜索（列优先+行索引非减）
    
    参数:
        distance_matrix: 二维numpy数组，形状为(m, n)，m行n列的距离矩阵
    
    返回:    
        path: 路径列表，格式为[(i0,j0), (i1,j1), ..., (in-1,jn-1)]，其中j从0到n-1递增，i非减
    """
    m, n = distance_matrix.shape  # m行n列
    if n == 0:
        return []
    
    # Step 1: 动态规划表初始化 (dp[j][i] = 第j列选第i行的最小累积距离)
    dp = np.full((n, m), np.inf)  # n列m行
    prev = np.full((n, m), -1, dtype=int)  # 记录前驱行索引
    
    # 第一列(j=0)初始化
    for i in range(m):
        dp[0][i] = distance_matrix[i][0]
    
    # Step 2: 填充动态规划表
    for j in range(1, n):  # 遍历列（必须从左到右）
        for i in range(m):  # 遍历当前列的行
            # 只能选择前一列中 ≤ 当前行i的行k
            min_prev = np.inf
            best_k = -1
            for k in range(i+1):  # k ≤ i（行索引非减约束）
                if dp[j-1][k] < min_prev:
                    min_prev = dp[j-1][k]
                    best_k = k
            if best_k != -1:
                dp[j][i] = min_prev + distance_matrix[i][j]
                prev[j][i] = best_k
    
    # Step 3: 回溯最优路径
    path = []
    # 从最后一列找到最小累积距离的行索引
    j = n - 1
    i = np.argmin(dp[j])
    path.append((i, j))
    
    # 回溯到第一列
    while j > 0:
        i = prev[j][i]
        j -= 1
        path.append((i, j))
    
    # 反转路径，从第0列到第n-1列
    path.reverse()
    return path


def dtw_align(path_a, path_b, radius = 5, data_set = mphoi()):
    with open(data_set.ground_truth) as f:
        data = json.load(f)
    input_a = zarr.open(path_a, mode='r')
    input_b = zarr.open(path_b, mode='r')

    caption_feature_dict = {}
    for video_id, ground_truth in data.items():
        caption_a = input_a[video_id][:][:,0,:]
        caption_b = input_b[video_id][:][:, 0, :]

        distance_matrix0 = cdist(caption_a, caption_b, metric='euclidean')

        path = constrained_dtw_path(distance_matrix0)
        indices = [p[0] for p in path]
        caption_c = caption_a[indices]

        distance_matrix = cdist(caption_c, caption_b, metric='euclidean')

        plt.figure(figsize=(10,8))
        plt.subplot(211)
        sns.heatmap(
            distance_matrix0,
            annot=False,
            fmt='.2f',
            cmap='YlGnBu',
            # xticklabels=labels,
            # yticklabels=labels,
            vmin=2,
            vmax=10
        )
        plt.subplot(212)
        sns.heatmap(
            distance_matrix,
            annot=False,
            fmt='.2f',
            cmap='YlGnBu',
            # xticklabels=labels,
            # yticklabels=labels,
            vmin=2,
            vmax=10
        )
        plt.title(video_id, fontsize=14)
        plt.tight_layout()
        save_path = os.path.join('./matrix_plot/bimanuals', video_id + '.png')
        plt.savefig(save_path)

        caption_feature_dict[video_id] = caption_c

    caption_feature_save_name = 'caption_features_doubao_aligned.zarr'
    caption_feature_save_path = os.path.join(data_set.output_dir, caption_feature_save_name)
    caption_feature_store = zarr.DirectoryStore(caption_feature_save_path)
    caption_feature_root = zarr.group(store = caption_feature_store)
    for key, value in caption_feature_dict.items():
        caption_feature_root.create_dataset(
                name = key,
                data = value,
                chunks = (1, 768),
                dtype = 'float32'
                )

    print('done')

@hydra.main(version_base= None, config_path='conf',config_name='DETR')
def main(args: DictConfig):
    # detect_on_VidHOI(args)
    GetVisualFeatureDETR(args=args)

# if __name__ == '__main__':
    # main()
    # CaptionGenerationWithDoubaoAPI(data_set=mphoi())
    # dataset = bimanualactions()
    # dataset.img2video()
    # caption_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_ground_truth_captions.json'
    # output_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_ground_truth_captions_downsampled.json'
    # Caption_downsampling(caption_path, output_path)
    # Label2Caption(data_set=bimanualactions())
    # ActionFeatureEncode(data_set=cad120())
    # ActionFeatureEncode(data_set=bimanualactions())


    # CaptionFeatureEncode(data_set=bimanualactions(), caption_path='data/BimanualActions/Real_time_captions.json', generate_caption=False)

    # path_a = 'data/BimanualActions/bimacs_derived_features/caption_features_doubao.zarr'
    # path_b = 'data/BimanualActions/bimacs_derived_features/caption_features_ground_truth.zarr'
    # dtw_align(path_a, path_b, data_set=bimanualactions())

    # Label2Caption(mphoi())

    # Label2Caption