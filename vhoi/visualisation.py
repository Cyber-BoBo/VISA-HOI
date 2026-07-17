import os
import cv2
import zarr
import json
import math
import torch
import numpy as np
# import seaborn as sns

from natsort import natsorted
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Dict
from itertools import groupby
from sklearn.metrics.pairwise import cosine_similarity


def create_label_bar(label_ids: list, bar_height: int = 30, bar_width: int = 5):
    """Create a numpy array that represents the video segmentation.

    Arguments:
        label_ids - Video segmentation represented as a list of label IDs.
        bar_height - Height (or number of rows) of the returned numpy array.
        bar_width - Width of individual segments in the desired plot. The returned array has len(label_ids) * bar_width
            columns.
    Returns:
        A numpy array of shape (bar_height, len(label_ids) * bar_width) containing a representation of video
        segmentation.
    """
    label_bar = np.empty([bar_height, bar_width * len(label_ids)])
    for i, label in enumerate(label_ids):
        label_bar[:, i * bar_width:(i + 1) * bar_width] = label
    return label_bar


def determine_xlabels_and_xticks_positions(labels: list, bar_width: int):
    """Simplify segmentation labelling in case of frame-wise segmentation.

    From a list of frame-level labels, extract the unique labels and determine x-axis positions to plot them.

    Arguments:
        labels - Video segmentation as a list of labels.
        bar_width - Width of a single segment bar in the expected plot.
    Returns:
        Two lists. The first one contains the unique labels in labels, and the second contain the x-axis position to
        place the labels in the final segmentation plot.
    """
    unique_labels, xticks, cumulative_length = [], [], 0
    for k, v in groupby(labels):
        unique_labels.append(k)
        num_frames = len(list(v))
        if xticks:
            xticks.append(cumulative_length + (num_frames // 3))
        else:
            xticks.append(num_frames // 3)
        xticks[-1] *= bar_width
        cumulative_length += num_frames
    return unique_labels, xticks


def plot_segmentation(target: list, *output, class_id_to_label: Dict[int, str], save_file: str = None,
                      bar_height: int = 30, bar_width: int = 2000, xlabels_type: str = 'label'):
    """Plot ground-truth and predicted segmentations.

    Arguments:
        target - A list containing the ground-truth label IDs.
        output - Output predictions to compare against the target. Each element is a list containing the predicted
            labels IDs.
        class_id_to_label - Dictionary mapping label IDs to label names.
        save_file - Optional file to write out segmentation plot.
        bar_height - Height of the bars drawn.
        bar_width - Width of the bars drawn.
        xlabels_type - One of 'label', 'id', or None.
    """
    bar_width = int(bar_width / len(target))
    num_classes = len(class_id_to_label)
    plt.figure(figsize=(num_classes, 1))
    grid_spec = mpl.gridspec.GridSpec(1 + len(output), 1)
    grid_spec.update(wspace=0.5, hspace=0.01)
    for plt_idx, label_ids in enumerate([target, *output]):
        ax = plt.subplot(grid_spec[plt_idx])
        label_bar = create_label_bar(label_ids, bar_height=bar_height, bar_width=bar_width)
        label_bar = label_bar.astype(np.int8)
        plt.imshow(label_bar, cmap=plt.get_cmap('tab20'), vmin=0, vmax=num_classes - 1)
        ax.tick_params(axis='both', which='both', length=0)
        xlabels, xticks = determine_xlabels_and_xticks_positions(label_ids, bar_width)
        ax.set_xticks(xticks)
        fontsize = 'small'
        if xlabels_type == 'labels':
            xlabels, fontsize = [class_id_to_label[label_id] for label_id in xlabels], 'x-small'
        elif xlabels_type == 'id':
            xlabels = [str(label_id) for label_id in xlabels]
        else:
            xlabels = []
        ax.set_xticklabels(xlabels, fontsize=fontsize, horizontalalignment='left')
        ax.set_yticklabels([])
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0, transparent=True)
    else:
        plt.show()
    plt.close()


def plot_training_curves(model_dirs, names, save_path):

    for name, model_dir in zip(names, model_dirs):
        checkpoint_file = os.path.join(model_dir, os.path.basename(model_dir) + '.tar')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        train_losses = [sum(epoch_losses) for _, epoch_losses in checkpoint['train_losses']]
        val_losses = [sum(epoch_losses) for _, epoch_losses in checkpoint['val_losses']]
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=name[0])
        plt.plot(range(1, len(val_losses) + 1), val_losses, label=name[1])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()



def list_images(directory, extensions=['.jpg', '.png']):
    """
    列出指定目录下所有图片文件的路径。
    
    :param directory: 要搜索的目录路径
    :param extensions: 要搜索的文件扩展名列表
    :return: 图片文件的路径列表
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        files.sort()
        # print(f"当前目录: {root}")
        # print("子目录:")
        # for dir in dirs:
            # print(f"  {dir}")
        # print("文件:")
        for file in files:
            # print(f"  {file}")
            if os.path.splitext(file)[1].lower() in extensions:
                image_files.append(os.path.join(root, file))
        # print("-" * 20)
    return image_files

def num2str(num,index):
    num_s = str(num)
    str_num = num_s
    for i in range(0,index-len(num_s)):
        str_num = '0' + str_num
    return str_num

def visualize_image(predict_path, dataset_name = 'mphoi', subject_id = 1, task_id = 1, take_id = 1):


    if dataset_name == 'bimanual':
        id2task = ['cooking','cooking_with_bowls','pouring','wiping','cereals','hard_drive','free_hard_drive','hammering','sawing']
        id2label = ['idle','approach','retreat','lift','place','hold','pour','cut','hammer','saw','stir','screw','drink','wipe']
        hand = ['left hand','right hand']

        label_path = './data/BimanualActions/bimacs_action_id_to_action_name.json'

        video_path = './data/BimanualActions/bimacs_videos'

        if task_id<=5:
            target_video = f'subject_{subject_id}-task_{task_id}_k_'+id2task[task_id-1]+f'-take_{take_id}'
            img_path = os.path.join(f'subject_{subject_id}', f'task_{task_id}_k_'+id2task[task_id-1], f'take_{take_id}')
            
        else:
            target_video = f'subject_{subject_id}-task_{task_id}_w_'+id2task[task_id-1]+f'-take_{take_id}'
            img_path = os.path.join(f'subject_{subject_id}', f'task_{task_id}_w_'+id2task[task_id-1], f'take_{take_id}')
        video_path = os.path.join(video_path, target_video)
            
        # 加载skeleton数据和boudding box数据
        data_bbs_path = './data/BimanualActions/bimacs_derived_features/bounding_boxes.zarr'
        data_hps_path = './data/BimanualActions/bimacs_derived_features/hands_pose.zarr'
        root_bbs = zarr.open(data_bbs_path, mode='r')
        root_hps = zarr.open(data_hps_path, mode='r')

        image_path = os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_raw_datas', img_path, 'rgb')

        # 物体边界框
        object_bbs = root_bbs[target_video]['objects'][:]
        # 人手边界框
        human1_bbs = root_bbs[target_video]['left_hand'][:]
        human2_bbs = root_bbs[target_video]['right_hand'][:]
        # 人体 skeleton
        human1_pose = root_hps[target_video]['left_hand'][:]
        human2_pose = root_hps[target_video]['right_hand'][:]

    elif dataset_name == 'mphoi':
        id2task = ['cheering','hair_cutting','co_working']
        id2label = ["sit","approach","retreat","lift","place","pour","drink","cheers","cut","dry","work","ask","solve"]
        hand = ['Human1','Human2']

        target_video = f'Subject{subject_id}-task_{task_id}_'+id2task[task_id-1]+f'-take_{take_id}'
        image_path = os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_raw_datas/MPHOI_rgb', 
                                f'Subject{subject_id}',f'task_{task_id}_'+id2task[task_id-1],f'take_{take_id}')

            
        # 加载skeleton数据和boudding box数据
        data_hbbs_path = './data/MPHOI/mphoi_derived_features/human_bounding_boxes.zarr'
        data_obbs_path = './data/MPHOI/mphoi_derived_features/object_bounding_boxes.zarr'
        data_hps_path = './data/MPHOI/mphoi_derived_features/human_pose.zarr'
        root_hbbs = zarr.open(data_hbbs_path, mode='r')
        root_obbs = zarr.open(data_obbs_path, mode='r')
        root_hps = zarr.open(data_hps_path, mode='r')


        # 物体边界框
        object_bbs = root_obbs[target_video]['objects'][:]

        human1_bbs = root_hbbs[target_video]['Human1'][:]
        human2_bbs = root_hbbs[target_video]['Human2'][:]

        human1_pose = root_hps[target_video]['Human1'][:]
        human2_pose = root_hps[target_video]['Human2'][:]

    # 读取目标样本文件夹下所有图片路径
    test = list_images(image_path)
    # # 文件路径列表排序
    sub_test = []
    for i in range(2,10):
        sub_test.append(test.pop(10*(i-1)+2))
    for i in range(2,10):
        test.insert(i,sub_test[i-2])

    # 读取Prediction结果
    Predictions = []
    with open(predict_path, mode='rb') as f:
        data = json.load(f)
        for video_id, hands_ground_truth in data.items():
            if target_video == video_id:
                for hand_id, hand_labels in hands_ground_truth.items():
                    for i in range(0,len(hand_labels)):
                        Prediction = []
                        Prediction.append(hand[int(hand_id)-1]+':'+id2label[hand_labels[i]])
                        Predictions.append(Prediction)
    Predictions = np.array(Predictions)
    l = Predictions.size
    Predictions = Predictions.reshape((2,int(l/2))).T

        


    # 图片保存路径
    # image_save_path = os.path.join('Analysis',dataset_name,'visualize_output',target_video)
    # if not os.path.exists(image_save_path):
    #     os.makedirs(image_save_path)


    # 绘图颜色设置
    ob_bbs_color = [(192,0,0),(255,0,0),(255,153,0),(255,255,0),(153,255,51),(51,204,51),(0,255,153),(51,204,204),(0,153,255),(51,102,255)]
    handpose_point_color = (0,0,192)


    # 视频输出设置
    
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    # cap_fps = 24
    # vid_size = (640,480)
    # output_name = target_video + '.mp4'
    # video = cv2.VideoWriter(os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Video', output_name), fourcc, cap_fps, vid_size)
    output_dir = os.path.join('//mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/HOI/My_Work/VISA-HOI/GitHOI/vis',target_video)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 绘图
    for i in range(0,len(test)-1):
        img = cv2.imread(test[i])
        # 绘制边界框
        # 物体
        #######################

        # for ob_index in range(0,len(object_bbs[i])):
        #     if object_bbs[i][ob_index][0] == object_bbs[i][ob_index][0]:
        #         cv2.rectangle(img, (int(object_bbs[i][ob_index][0]), int(object_bbs[i][ob_index][1])), (int(object_bbs[i][ob_index][2]), int(object_bbs[i][ob_index][3])), ob_bbs_color[ob_index], 2)
        # 左右手
        # cv2.rectangle(img, (int(human1_bbs[i][0]), int(human1_bbs[i][1])), (int(human1_bbs[i][2]), int(human1_bbs[i][3])), ob_bbs_color[-1], 2)
        # cv2.rectangle(img, (int(human2_bbs[i][0]), int(human2_bbs[i][1])), (int(human2_bbs[i][2]), int(human2_bbs[i][3])), ob_bbs_color[-1], 2)
        # for hp_index in range(0,len(human1_pose[i])):
        #     if not math.isnan(human1_pose[i][hp_index][0]): 
        #         cv2.circle(img, (int(human1_pose[i][hp_index][0]),int(human1_pose[i][hp_index][1])), 2, handpose_point_color, 2)
        #     if not math.isnan(human2_pose[i][hp_index][0]): 
        #         cv2.circle(img, (int(human2_pose[i][hp_index][0]),int(human2_pose[i][hp_index][1])), 2, handpose_point_color, 2)
        cv2.putText(img,Predictions[i,0], (10,45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), thickness = 2)
        cv2.putText(img,Predictions[i,1], (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), thickness = 2)
        output_path = os.path.join(output_dir,num2str(i,3)+'.png')
        cv2.imwrite(output_path,img)
        ###########################
        cv2.imshow('HOI',img)
        cv2.waitKey(1)
        # video.write(img)
    # video.release

def visualize_skeletons(data_dir):
    with open(data_dir) as f:
        data = json.load(f)
    left_pose = np.array(data['left_hand'])
    right_pose = np.array(data['right_hand'])

    plt.figure()
    plt.scatter(
        left_pose[:,0],
        left_pose[:,1],
        color = 'blue'
    )
    plt.scatter(
        right_pose[:,0],
        right_pose[:,1],
        color = 'red'
    )
    plt.show()


def visualize_bimanual_video(predict_path, subject_id = 1, task_id = 1, take_id = 1):

    id2task = ['cooking','cooking_with_bowls','pouring','wiping','cereals','hard_drive','free_hard_drive','hammering','sawing']
    id2label = ['idle','approach','retreat','lift','place','hold','pour','cut','hammer','saw','stir','screw','drink','wipe']
    hand = ['left hand','right hand']

    # label_path = './data/BimanualActions/bimacs_action_id_to_action_name.json'

    input_video_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_videos'
    output_video_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Video'

    if task_id<=5:
        target_video = f'subject_{subject_id}-task_{task_id}_k_'+id2task[task_id-1]+f'-take_{take_id}'  
    else:
        target_video = f'subject_{subject_id}-task_{task_id}_w_'+id2task[task_id-1]+f'-take_{take_id}'
    
    video_file = target_video + '.mp4'
    input_video = os.path.join(input_video_path, video_file)
    output_video = os.path.join(output_video_path, video_file)
        

    # 读取Prediction结果
    Predictions = []
    with open(predict_path, mode='rb') as f:
        data = json.load(f)
        for video_id, hands_ground_truth in data.items():
            if target_video == video_id:
                for hand_id, hand_labels in hands_ground_truth.items():
                    for i in range(0,len(hand_labels)):
                        Prediction = []
                        Prediction.append(hand[int(hand_id)-1]+':'+id2label[hand_labels[i]])
                        Predictions.append(Prediction)
    Predictions = np.array(Predictions)
    l = Predictions.size
    Predictions = Predictions.reshape((2,int(l/2))).T

    # 视频输入
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f'无法打开视频文件：{input_video}')

    ori_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 视频输出设置
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    out_fps = 15
    video = cv2.VideoWriter(output_video, fourcc, out_fps, (width,height))
    
    
    # 绘图设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 1
    font_thickness = 2


    # 绘图
    for frame_idx in range(len(Predictions)):
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, Predictions[frame_idx,0], (10,30), font, font_size, (0,0,255), thickness = font_thickness)
        cv2.putText(frame, Predictions[frame_idx,1], (10,60), font, font_size, (255,0,0), thickness = font_thickness)

        video.write(frame)
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    # video.release


def visualize_mphoi_video(subject_id = 1, task_id = 1, take_id = 1):

    id2task = ['cheering','hair_cutting','co_working']
    id2label = ["sit","approach","retreat","lift","place","pour","drink","cheers","cut","dry","work","ask","solve"]
    hand = ['Human1','Human2']

    target_video = f'Subject{subject_id}-task_{task_id}_'+id2task[task_id-1]+f'-take_{take_id}'

    # label_path = './data/BimanualActions/bimacs_action_id_to_action_name.json'

    input_image_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Image_for_video'
    output_video_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Video'
    
    video_file = target_video + '.mp4'
    input_frames = os.path.join(input_image_path, target_video)
    output_video = os.path.join(output_video_path, video_file)

    # 图像文件夹输入
    images = list_images(input_frames)

    # 视频输出设置
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    out_fps = 30
    video = cv2.VideoWriter(output_video, fourcc, out_fps, (3840,2160))
    

    # 绘图
    for frame_idx in range(len(images)):
        frame = cv2.imread(images[frame_idx])
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()


def visualize_HRI_video(task):

    target_video = f'HRI_{task}.mp4'
    input_image_path = f'/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/HOI/My_Work/OmniHOI/GitHOI/image_test/output/Videos/{task}/video_1'
    output_video_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Video'

    images = natsorted(list_images(input_image_path))
    output_video = os.path.join(output_video_path, target_video)

    # 视频输出设置
    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
    out_fps = 30
    video = cv2.VideoWriter(output_video, fourcc, out_fps, (640,480))
    

    # 绘图
    for frame_idx in range(len(images)):
        frame = cv2.imread(images[frame_idx])
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()


# def visualize_data_preprocesssed(dataset_name = 'mphoi', subject_id = 1, task_id = 1, take_id = 1):


#     if dataset_name == 'bimanual':
#         id2task = ['cooking','cooking_with_bowls','pouring','wiping','cereals','hard_drive','free_hard_drive','hammering','sawing']
#         id2label = ['idle','approach','retreat','lift','place','hold','pour','cut','hammer','saw','stir','screw','drink','wipe']
#         hand = ['left hand','right hand']

#         label_path = './data/BimanualActions/bimacs_action_id_to_action_name.json'

#         video_path = '/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/BimanualActions/bimacs_videos'

#         if task_id<=5:
#             video_name = f'subject_{subject_id}-task_{task_id}_k_'+id2task[task_id-1]+f'-take_{take_id}.mp4'
            
#         else:
#             video_name = f'subject_{subject_id}-task_{task_id}_w_'+id2task[task_id-1]+f'-take_{take_id}.mp4'
#         video_path = os.path.join(video_path, video_name)
            
#         # 加载skeleton数据和boudding box数据
#         data_bbs_path = './data/BimanualActions/bimacs_derived_features/bounding_boxes.zarr'
#         data_hps_path = './data/BimanualActions/bimacs_derived_features/hands_pose.zarr'
#         root_bbs = zarr.open(data_bbs_path, mode='r')
#         root_hps = zarr.open(data_hps_path, mode='r')


#         # 物体边界框
#         object_bbs = root_bbs[target_video]['objects'][:]
#         # 人手边界框
#         human1_bbs = root_bbs[target_video]['left_hand'][:]
#         human2_bbs = root_bbs[target_video]['right_hand'][:]
#         # 人体 skeleton
#         human1_pose = root_hps[target_video]['left_hand'][:]
#         human2_pose = root_hps[target_video]['right_hand'][:]

#     elif dataset_name == 'mphoi':
#         id2task = ['cheering','hair_cutting','co_working']
#         id2label = ["sit","approach","retreat","lift","place","pour","drink","cheers","cut","dry","work","ask","solve"]
#         hand = ['Human1','Human2']

#         target_video = f'Subject{subject_id}-task_{task_id}_'+id2task[task_id-1]+f'-take_{take_id}'
#         image_path = os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Dataset/MPHOI-72/mphoi_raw_datas/MPHOI_rgb', 
#                                 f'Subject{subject_id}',f'task_{task_id}_'+id2task[task_id-1],f'take_{take_id}')

            
#         # 加载skeleton数据和boudding box数据
#         data_hbbs_path = './data/MPHOI/mphoi_derived_features/human_bounding_boxes.zarr'
#         data_obbs_path = './data/MPHOI/mphoi_derived_features/object_bounding_boxes.zarr'
#         data_hps_path = './data/MPHOI/mphoi_derived_features/human_pose.zarr'
#         root_hbbs = zarr.open(data_hbbs_path, mode='r')
#         root_obbs = zarr.open(data_obbs_path, mode='r')
#         root_hps = zarr.open(data_hps_path, mode='r')


#         # 物体边界框
#         object_bbs = root_obbs[target_video]['objects'][:]

#         human1_bbs = root_hbbs[target_video]['Human1'][:]
#         human2_bbs = root_hbbs[target_video]['Human2'][:]

#         human1_pose = root_hps[target_video]['Human1'][:]
#         human2_pose = root_hps[target_video]['Human2'][:]

#     # 读取目标样本文件夹下所有图片路径
#     test = list_images(image_path)
#     # 文件路径列表排序
#     # sub_test = []
#     # for i in range(2,10):
#     #     sub_test.append(test.pop(10*(i-1)+2))
#     # for i in range(2,10):
#     #     test.insert(i,sub_test[i-2])



#     # 图片保存路径
#     image_save_path = os.path.join('Output/Analysis',dataset_name,'visualize_output',target_video)
#     if not os.path.exists(image_save_path):
#         os.makedirs(image_save_path)


#     # 绘图颜色设置
#     ob_bbs_color = [(192,0,0),(255,0,0),(255,153,0),(255,255,0),(153,255,51),(51,204,51),(0,255,153),(51,204,204),(0,153,255),(51,102,255)]
#     handpose_point_color = (0,0,192)


#     # 视频输出设置
    
#     # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
#     # cap_fps = 24
#     # vid_size = (640,480)
#     # output_name = target_video + '.mp4'
#     # video = cv2.VideoWriter(os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Video', output_name), fourcc, cap_fps, vid_size)
#     output_dir = os.path.join('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/Papers/MyWork/VISA-HOI/Image_for_video',target_video)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 绘图
#     for i in range(0,len(test)-1):
#         img = cv2.imread(test[i])
#         # 绘制边界框
#         # 物体
#         #######################

#         # for ob_index in range(0,len(object_bbs[i])):
#         #     if object_bbs[i][ob_index][0] == object_bbs[i][ob_index][0]:
#         #         cv2.rectangle(img, (int(object_bbs[i][ob_index][0]), int(object_bbs[i][ob_index][1])), (int(object_bbs[i][ob_index][2]), int(object_bbs[i][ob_index][3])), ob_bbs_color[ob_index], 2)
#         # # 左右手
#         # cv2.rectangle(img, (int(human1_bbs[i][0]), int(human1_bbs[i][1])), (int(human1_bbs[i][2]), int(human1_bbs[i][3])), ob_bbs_color[-1], 2)
#         # cv2.rectangle(img, (int(human2_bbs[i][0]), int(human2_bbs[i][1])), (int(human2_bbs[i][2]), int(human2_bbs[i][3])), ob_bbs_color[-1], 2)
#         # for hp_index in range(0,len(human1_pose[i])):
#         #     if not math.isnan(human1_pose[i][hp_index][0]): 
#         #         cv2.circle(img, (int(human1_pose[i][hp_index][0]),int(human1_pose[i][hp_index][1])), 2, handpose_point_color, 2)
#         #     if not math.isnan(human2_pose[i][hp_index][0]): 
#         #         cv2.circle(img, (int(human2_pose[i][hp_index][0]),int(human2_pose[i][hp_index][1])), 2, handpose_point_color, 2)
#         cv2.putText(img,Predictions[i,0], (10,230), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), thickness = 15)
#         cv2.putText(img,Predictions[i,1], (10,480), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), thickness = 15)
#         output_path = os.path.join(output_dir,num2str(i,3)+'.png')
#         cv2.imwrite(output_path,img)
#         ###########################
#         # cv2.imshow('HOI',img)
#         # cv2.waitKey(5)
#         # video.write(img)
#     # video.release

if __name__ == '__main__':
    # model_dirs = ['outputs/bimanual/GitHOI/cTrue_mtv3_nl1_hs64_e40_bs32_lr0.001_0.1_2']
    # names = [['train loss 2', 'val loss 2']]
    # save_path = './Output/Analysis/curves/loss1_on_Bimanual.png'
    # plot_training_curves(model_dirs, names, save_path)

    # visualize_skeletons('image_test/output/hand_skeleton.json')


    dataset_name = 'bimanual' #bimanual
    subject_id = 1
    task_id = 8
    take_id = 3
    # visualize_mphoi_video(subject_id,task_id,take_id)


    predict_path = './Output/Predict/bimanual/sa/sa_M3HOI.json'
    visualize_image(predict_path, dataset_name, subject_id, task_id, take_id)
    # predict_path = './Output/Predict/bimanual/VISA-HOI/outputs/our_sa.json'
    # visualize_bimanual_video(predict_path,subject_id,task_id,take_id)


    # task = 'place'
    # visualize_HRI_video(task)


    # plot_similarity()