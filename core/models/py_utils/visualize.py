# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import os
import torch.nn as nn


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=20):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    
    keep_inds  = (topk_scores[0] > 0.1)
    topk_inds = topk_inds[0][keep_inds].view(batch,-1)
    topk_scores = topk_scores[0][keep_inds].view(batch,-1)
    

    out_s = torch.zeros(scores.shape[0]*scores.shape[1]*scores.shape[2]*scores.shape[3])
    out_s[topk_inds] = 1
    
    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    
    
#     import pdb
#     pdb.set_trace()
    out_s = out_s.view(batch,scores.shape[1],scores.shape[2],scores.shape[3])
    
    return out_s,topk_scores

def visualize(image, tl_heat, br_heat,rand_name=0):
    outDir = "/sujh/code/CornerNet_Lite/visual/"
    
    if not os.path.exists(os.path.join(outDir,"tl_heatmap")):
        os.makedirs(os.path.join(outDir,"tl_heatmap"))
    
    if not os.path.exists(os.path.join(outDir,"br_heatmap")):
        os.makedirs(os.path.join(outDir,"br_heatmap"))
    if rand_name==0:
        rand_name = str(np.random.randint(1,100000))
    else:
        rand_name = str(rand_name)
    # image-size = [2, 3, 800, 800]
    # 这个图片的原始尺寸下采样4倍，就变成heatmap的维度，所以测试时不一定是128
    # [2, 15, 100, 100]，这个是不定的，看图片大小
    tl_heat = torch.sigmoid(tl_heat)
    # [2, 15, 288, 512]
    br_heat = torch.sigmoid(br_heat)
    
    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=5)
    br_heat = _nms(br_heat, kernel=5)
    
    #inds表示在w*h里面的位置
    tl_heat,tl_scores = _topk(tl_heat, K=80)
#     br_heat,br_scores = _topk(br_heat, K=80)
    
    K = tl_scores.shape[1]
#     print("len tl scores:{}".format(K))
    if K<10:
        K = 10
        tl_heat,tl_scores = _topk(tl_heat, K=K)
    br_heat,br_score = _topk(br_heat, K=K)
    
    
#     print("tl score, num:{}, mean:{}, max:{}, min:{}".format(len(tl_scores),torch.mean(tl_scores),torch.max(tl_scores),torch.min(tl_scores)))
#     print("br score, num:{}, mean:{}, max:{}, min:{}".format(len(br_scores),torch.mean(br_scores),torch.max(br_scores),torch.min(br_scores)))
    
    # 这个colors是一个list，shape为(15, 1, 1, 3)，7是类别数，1，1，3是随机random的
    # 这个作用就是给每个类定制了专属的随机生成的颜色
    colors = [((np.random.random((1, 1, 3)) * 0.6 + 0.4)*255).astype(np.uint8)\
               for _ in range(tl_heat.shape[1])]
    # tl_heat[0] size = [7, 288, 512]
    # 取走第一个batch的特征，配上颜色
    # tl_hm、br_hm的维度均是[h, w, 3]
    tl_hm = _gen_colormap(tl_heat[0].detach().cpu().numpy(), colors)
    br_hm = _gen_colormap(br_heat[0].detach().cpu().numpy(), colors)
    # 标准差和均值
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    # 为rgb的图片，每通道乘上标准差加上均值，相当于每通道分配一个数字
    img = (image[0].detach().cpu().numpy() * std + mean) * 255
    # 再把图片transpose成标准的样子
    img = img.astype(np.uint8).transpose(1, 2, 0)

    tl_blend = _blend_img(img, tl_hm)
    br_blend = _blend_img(img, br_hm)
    
    tl_path = os.path.join(outDir,"tl_heatmap/"+rand_name+".jpg")
    br_path = os.path.join(outDir,"br_heatmap/"+rand_name+".jpg")
    cv2.imwrite(tl_path, tl_blend)
    cv2.imwrite(br_path, br_blend)
#     cv2.imwrite("/sujh/code/CornerNet_Lite/visual/heatmap/source.jpg", img)
#     print("~~~save heatmaps OK!")

def _gen_colormap(heatmap, colors):
    # 这个heatmap的维度是[7, 288, 512]
    num_classes = heatmap.shape[0]
    h, w = heatmap.shape[1], heatmap.shape[2]
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        # np.maximum是两个输入进行对比，每次谁大就挑谁的，维度要一致
        # color_map维度[h, w, 3]
        # heatmap[i, :, :, np.newaxis]维度[h, w, 1]
        # colors[i]维度[1, 1, 3]
        # 最终右边这一长串其实是0-255的整型数字
        # 接着循环类别次，color_map一直更新，每次挑maximum的
        color_map = np.maximum(
            color_map, (heatmap[i, :, :, np.newaxis] * colors[i]).astype(np.uint8))
    return color_map


def _blend_img(back, fore, trans=0.7):
    '''
    back = img-->[h*4, w*4, 3]
    fore = tl_hm-->[h, w, 3]
    '''
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
        fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
        fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    ret = (back * (1. - trans) + fore * trans).astype(np.uint8)
    # 别越界了,ret的大小就是原图的大小
    ret[ret > 255] = 255
    return ret
    
def save_feature_to_img(features, source_img):
    outDir = "/sujh/code/CornerNet_Lite/visual/feature/"
    rand_name = str(np.random.randint(1,100000))
    out_f_path = os.path.join(outDir,"feature_"+rand_name+".jpg")
    out_i_path = os.path.join(outDir,"img_"+rand_name+".jpg")
    '''
    features:[batch, 256, 128, 128]
    '''
    # 只取一张图的第一个通道
    # [1, 1, 128, 128]
    feature = features[:1,0,:,:]
    # [128, 128]
    feature = feature.view(feature.shape[1],feature.shape[2])
    #to numpy
    feature = feature.data.numpy()
    #use sigmod to [0,1]
    feature = 1.0/(1+np.exp(-1*feature))
    # to [0,255]
    feature = np.round(feature*255)
    cv2.imwrite(out_f_path,feature)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    img = (source_img.detach().cpu().numpy() * std + mean) * 255
    img = img.astype(np.uint8).transpose(1, 2, 0)
    cv2.imwrite(out_i_path,img)
 

def visualize_feature(features, image,sigmod=False):
    outDir = "/sujh/code/CornerNet_Lite/visual/feature/"
    rand_name = str(np.random.randint(1,100000))
    out_f_path = os.path.join(outDir,"feature_"+rand_name+".jpg")
    out_i_path = os.path.join(outDir,"img_"+rand_name+".jpg")
    '''
    features:[batch, 256, 128, 128]
    '''
    # 只取一张图的第一个通道
    # [1, 1, 128, 128]
    feature = features[:1,0,:,:]
    # [128, 128]
    feature = feature.view(feature.shape[1],feature.shape[2])
    #to numpy
    feature = feature.detach().cpu().numpy()
    if sigmod:
        #use sigmod to [0,1]
        feature = 1.0/(1+np.exp(-1*feature))
    # to [0,255]
    feature = np.round(feature*255)
#     cv2.imwrite(out_f_path,feature)
    feature = _gen_maskmap(feature)
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    img = (image[0].detach().cpu().numpy() * std + mean) * 255
    # 再把图片transpose成标准的样子
    img = img.astype(np.uint8).transpose(1, 2, 0)
#     import pdb
#     pdb.set_trace()
    outImg = _blend_img(img, feature)
    
    cv2.imwrite(out_i_path,outImg)
     

def _gen_maskmap(feature):
    # feature[64, 64]
    h, w = feature.shape[0], feature.shape[1]
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if feature[i][j]>128:
                color_map[i][j] = [255,255,255]
    return color_map

def draw_proposals(image,bboxes,name=1):
    outDir = "/data/result/MKD-NET-voc-pro/"
#     outDir = "/sujh/code/CornerNet_Lite/visual/proposals/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)
            
    for bbox in bboxes:
        color = np.random.random((3, ))
        color = (color * 255).astype(np.int32).tolist()
    
        bbox = bbox[0:4].astype(np.int32)

        cv2.rectangle(image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color, 2
        )
    
    debug_file = os.path.join(outDir, "{}.jpg".format(name))
    cv2.imwrite(debug_file, image)
