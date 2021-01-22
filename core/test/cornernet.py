#coding:utf-8
import os
import cv2
import json
import numpy as np
import torch
import json

from tqdm import tqdm

from ..utils import Timer
from ..vis_utils import draw_bboxes
from ..sample.utils import crop_image
from ..external.nms import soft_nms, soft_nms_merge
from ..models.py_utils.visualize import draw_proposals

def rescale_dets_(detections, ratios, borders, sizes):
    #[开始：结尾：步长]
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs    /= ratios[:, 1][:, None, None]
    ys    /= ratios[:, 0][:, None, None]
    xs    -= borders[:, 2][:, None, None]
    ys    -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)

def decode(nnet, images, K, ae_threshold=0.5, kernel=3, num_dets=1000):
    multi_detections = nnet.test([images], ae_threshold=ae_threshold, test=True, K=K, kernel=kernel, num_dets=num_dets)
    multi_rs = []
    for detections in multi_detections:
        multi_rs.append(detections.data.cpu().numpy())
 
    return multi_rs

#added by su
def analyRs(jsonPath):
    fp =  open(jsonPath)
    objs = json.load(fp)
    
    print(objs[0])
    
    t_r = {}
    for i,obj in enumerate(objs):
        imgId = obj['image_id']
        ctId = obj['category_id']
        score = obj['score']
        if not imgId in t_r.keys():
            t_r[imgId] = [0,0,0,1]
        
        t_r[imgId][0] +=1
        t_r[imgId][1] +=score
        t_r[imgId][2] =max(t_r[imgId][2],score)
        t_r[imgId][3] =min(t_r[imgId][2],score)
        
    
    print(t_r)

#cornernet->cornernet_inference->decode->modules.test->utils.decode
def cornernet(db, nnet, result_dir, debug=False, decode_func=decode):
    print("db split:", db.split)
#     debug_dir = os.path.join(result_dir, "debug")
    debug_dir = "/data/result/MKD-NET-voc/"
#     debug_dir = "/home/disk1/jhsu/result/MKD-NET-voc"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    db_inds = db.db_inds[:500] if debug else db.db_inds

#     print("image ids:",db._image_ids[0:50])

    num_images = db_inds.size
    categories = db.configs["categories"]
    
#     out_meta_file="/sujh/result/MKD-Net-64.txt"
#     out_fp = open(out_meta_file,"w")
    

    timer = Timer()
    top_bboxes = {}
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
#     for ind in range(0,num_images):
        db_ind = db_inds[ind]

        image_id   = db.image_ids(db_ind)
        image_path = db.image_path(db_ind)
        image      = cv2.imread(image_path)

        timer.tic()
        top_bboxes[image_id] = cornernet_inference(db, nnet, image,ind=ind,image_id=image_id)
        timer.toc()
        
        if debug:
#         if False:
#         if False:
            image_path = db.image_path(db_ind)
            image      = cv2.imread(image_path)
            bboxes     = {
                db.cls2name(j): top_bboxes[image_id][j] 
                for j in range(1, categories + 1)
            }
            
            image      = draw_bboxes(image, bboxes)
            debug_file = os.path.join(debug_dir, "{}.jpg".format(db_ind))
            cv2.imwrite(debug_file, image)
    print('average time: {}'.format(timer.average_time))


    result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)

    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    #call the evaluate function in core/dbs/coco.py
    db.evaluate(result_json, cls_ids, image_ids)
    return 0

def cornernet_inference(db, nnet, image, decode_func=decode,ind=1,image_id=0):
    #保留通道得分的top k个点
    K             = db.configs["top_k"]
    #if tl_tag - br_tag > ae, reject
    ae_threshold  = db.configs["ae_threshold"]
    #kernel size of nms on heatmaps
    nms_kernel    = db.configs["nms_kernel"]
    #在K*K个框里面，每张图片保留num_dets数目
    num_dets      = db.configs["num_dets"]
    test_flipped  = db.configs["test_flipped"]

    input_size    = db.configs["input_size"]
    output_size   = db.configs["output_sizes"][0]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]

    height, width = image.shape[0:2]

#     height_scale  = 2*(input_size[0] + 1) // output_size[0]
#     width_scale   = 2* (input_size[1] + 1) // output_size[1]
    
    height_scale  = (input_size[0] + 1) // output_size[0]
    width_scale   = (input_size[1] + 1) // output_size[1]

    im_mean = torch.cuda.FloatTensor(db.mean).reshape(1, 3, 1, 1)
    im_std  = torch.cuda.FloatTensor(db.std).reshape(1, 3, 1, 1)

    detections = []
    #multi scales
    for scale in scales:
        new_height = int(height * scale)
        new_width  = int(width * scale)
        
#         new_height = 415
#         new_width = 415
        scale = new_height/height
        new_center = np.array([new_height // 2, new_width // 2])

        #保证+1后能被8整除
        inp_height = new_height | 127
        inp_width  = new_width  | 127

        images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios  = np.zeros((1, 2), dtype=np.float32)
        ratios_s  = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes   = np.zeros((1, 2), dtype=np.float32)

        #测试时，图片没有resize到511*511，而是直接除以8
        out_height, out_width = (inp_height + 1) // height_scale, (inp_width + 1) // width_scale
        height_ratio = out_height / inp_height
        width_ratio  = out_width  / inp_width
        
        #多尺度裁剪
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

        resized_image = resized_image / 255.

        #cv2.imread是b,g,r模式
        images[0]  = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0]   = [int(height * scale), int(width * scale)]
        ratios[0]  = [height_ratio, width_ratio]
        ratios_s[0]  = [height_ratio/2, width_ratio/2]
        
        if test_flipped:
            images  = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images  = torch.from_numpy(images).cuda()
        images -= im_mean
        images /= im_std

        multi_dets = decode_func(nnet, images, K, ae_threshold=ae_threshold, kernel=nms_kernel, num_dets=num_dets)
       
        for t_i,dets in enumerate(multi_dets):
            if test_flipped:
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
                dets = dets.reshape(1, -1, 8)

            
            if t_i<1:
                rescale_dets_(dets, ratios, borders, sizes)
            else:
                rescale_dets_(dets, ratios_s, borders, sizes)
            dets[:, :, 0:4] /= scale
            detections.append(dets)

    detections = np.concatenate(detections, axis=1)

    classes    = detections[..., -1]
    classes    = classes[0]
    detections = detections[0]

    # reject detections with negative scores
    keep_inds  = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes    = classes[keep_inds]
    
#     out_meta_file="/sujh/result/MKD-Net-64.txt"
#     out_fp = open(out_meta_file,"a+")
#     
#     #added,out logs
#     for b_i in range(classes.shape[0]):
#         cat_id = int(classes[b_i] + 1)
#         category_name = db.cls2name(cat_id)
#         xmin = float(detections[b_i][0])
#         ymin = float(detections[b_i][1])
#         xmax = float(detections[b_i][2])
#         ymax = float(detections[b_i][3])
#         score = float(detections[b_i][4])
#         tstr= ("{} {} {} {} {} {} {}\n".format(int(image_id[:-4]),cat_id-1,score,xmin,ymin,xmax,ymax))
#         out_fp.write(tstr)
#         
#     out_fp.close()
    
#     draw_proposals(image, detections,ind)
    
    #bboxes, scores, tl_scores, br_scores, clses
    top_bboxes = {}
    t_boxes = []
    for j in range(categories):
        keep_inds = (classes == j)
     
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        
        pr_len = len(top_bboxes[j + 1])
        if merge_bbox:
            soft_nms_merge(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
        else:
            soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]
        
        af_len = len(top_bboxes[j + 1])
#         if pr_len>0:
#             print("pre:{}, af:{}".format(pr_len,af_len))
        
        t_boxes.extend(top_bboxes[j + 1])
    
#     draw_proposals(image, t_boxes,ind)
    
    scores = np.hstack([top_bboxes[j][:, -1] for j in range(1, categories + 1)])
    
#     out_meta_file="/sujh/result/MKD-Net-64.txt"
#     out_fp = open(out_meta_file,"a+")
#     
#     for j in range(1, categories + 1):
# #         category_name = db.cls2name(j)
#         for t_box in top_bboxes[j]:
#             if t_box[4]<0.5:
#                 continue
#             xmin = float(t_box[0])
#             ymin = float(t_box[1])
#             xmax = float(t_box[2])
#             ymax = float(t_box[3])
#             score = float(t_box[4])
#             tstr= ("{} {} {} {} {} {} {}\n".format(int(image_id[:-4]),j-1,score,xmin,ymin,xmax,ymax))
#             out_fp.write(tstr)
# 
#     out_fp.close()
    
    
    if len(scores) > max_per_image:
        kth    = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, categories + 1):
            keep_inds     = (top_bboxes[j][:, -1] >= thresh)
            top_bboxes[j] = top_bboxes[j][keep_inds]
    return top_bboxes
