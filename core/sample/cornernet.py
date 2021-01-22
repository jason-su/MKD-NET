#coding:utf-8
import cv2
import math
import numpy as np
import torch
import os

from .utils import random_crop, draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

#read train samples
def cornernet(system_configs, db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 512
    fp_scales=[1,2]

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = []
    br_heatmaps = []
    #存offset
    tl_regrs = []
    br_regrs = []
    #存每个角点的位置
    tl_tags = []
    br_tags = []
    tag_lens = [] #存放每张图片的目标数量
    
    for  _fpscale in fp_scales:
        tl_heatmaps.append(np.zeros((batch_size, categories, output_size[0]//_fpscale, output_size[1]//_fpscale), dtype=np.float32))
        br_heatmaps.append(np.zeros((batch_size, categories, output_size[0]//_fpscale, output_size[1]//_fpscale), dtype=np.float32))
    
        tl_regrs.append(np.zeros((batch_size, max_tag_len, 2), dtype=np.float32))
        br_regrs.append(np.zeros((batch_size, max_tag_len, 2), dtype=np.float32))
        tl_tags.append(np.zeros((batch_size, max_tag_len), dtype=np.int64))
        br_tags.append(np.zeros((batch_size, max_tag_len), dtype=np.int64))
        tag_lens.append(np.zeros((batch_size, ), dtype=np.int32))
        
    #表示每张图片的目标个数，例如tag_masks[1]=[1,1,1,1,1,0,0...,0]表示第一张图片有5个目标
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    
    #定义数组：size:batch_size*class num(15)
    class_gt = np.zeros((batch_size,categories),dtype=np.float32)
    

    db_size = db.db_inds.size
    #b_ind represent batch_size_id
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)
        
        if not os.path.exists(image_path):
            print("not exist image:",image_path)
            
        # reading detections
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)

        image, detections = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            image[:] = image[:, ::-1, :]
            width    = image.shape[1]
            detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
            normalize_(image, db.mean, db.std)
        images[b_ind] = image.transpose((2, 0, 1))

        for s_i, _fpscale in enumerate(fp_scales):
            output_size_u = [output_size[0]//_fpscale,output_size[1]//_fpscale]
            width_ratio = output_size_u[1] / input_size[1]
            height_ratio = output_size_u[0] / input_size[0]
            
        
            for ind, detection in enumerate(detections):
                category = int(detection[-1]) - 1
                class_gt[b_ind][category]=1
    
                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]
    
                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)
    
                xtl = int(fxtl)
                ytl = int(fytl)
                xbr = int(fxbr)
                ybr = int(fybr)
                
                if gaussian_bump:
                    #惩罚的减少量由非标准化的2D高斯来计算
                    width  = detection[2] - detection[0]
                    height = detection[3] - detection[1]
    
                    width  = math.ceil(width * width_ratio)
                    height = math.ceil(height * height_ratio)
    
                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad
    
                    #edit by su
                    draw_gaussian(tl_heatmaps[s_i][b_ind, category], [xtl, ytl], radius)
                    draw_gaussian(br_heatmaps[s_i][b_ind, category], [xbr, ybr], radius)
                    
                else:
                    #1 truth
                    #edit by su
                    tl_heatmaps[s_i][b_ind, category, ytl, xtl] = 1
                    br_heatmaps[s_i][b_ind, category, ybr, xbr] = 1
    
                #b_ind: batch_ind
                #tag_ind: tag_ind
                tag_ind = tag_lens[s_i][b_ind]
                #the offset of point
                tl_regrs[s_i][b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
                br_regrs[s_i][b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
                
                #top-lef point position:
                #y*width + x,用一个值来表示角点的位置,1,2,...,width*height
                tl_tags[s_i][b_ind, tag_ind] = ytl * output_size_u[1] + xtl
                #bottom-right point position
                br_tags[s_i][b_ind, tag_ind] = ybr * output_size_u[1] + xbr
                tag_lens[s_i][b_ind] += 1
                
    
    
    for b_ind in range(batch_size):
        tag_len = tag_lens[0][b_ind]
        tag_masks[b_ind, :tag_len] = 1
    
    images      = torch.from_numpy(images)
    tl_heatmaps = [torch.from_numpy(_tl_heatmaps) for _tl_heatmaps in tl_heatmaps]
    br_heatmaps = [torch.from_numpy(_br_heatmaps) for _br_heatmaps in br_heatmaps]
    tl_regrs    = [torch.from_numpy(_tl_regrs) for _tl_regrs in tl_regrs]
    br_regrs    = [torch.from_numpy(_br_regrs) for _br_regrs in br_regrs]
    tl_tags     = [torch.from_numpy(_tl_tags) for _tl_tags in tl_tags]
    br_tags     = [torch.from_numpy(_br_tags) for _br_tags in br_tags]
    tag_masks   = torch.from_numpy(tag_masks)
    class_gt = torch.from_numpy(class_gt)

    return {
        "xs": [images],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags,class_gt]
    }, k_ind
