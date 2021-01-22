#coding:utf-8
import torch
import torch.nn as nn

from .utils import _tranpose_and_gather_feat

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _off_loss(off, gt_off, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_off)

    off    = off[mask]
    gt_off = gt_off[mask]
    
    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss

def _focal_loss_mask(preds, gt, mask):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    pos_mask = mask[pos_inds]
    neg_mask = mask[neg_inds]

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * pos_mask
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights * neg_mask

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

#p_pos*(1-p_pos)**2
#w*(1-p_neg)*p_neg**2
def _focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        #nelement:the number of list
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


'''
added 2019/11/09
'''
#h=-yt * logp - (1-yt)log(1-p)
def _cross_entropy_loss(pred, gt):
#     import pdb
#     pdb.set_trace()
        
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    loss = 0
    NEAR_0 =1e-10

    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = -torch.log(pos_pred + NEAR_0) 
    neg_loss = -torch.log(1 - neg_pred + NEAR_0)
    
#         pos_loss[pos_loss==float("inf")] =0
#         neg_loss[neg_loss==float("inf")] =0
    
    
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    loss = loss + (pos_loss + neg_loss)/gt.nelement()
    
    if torch.isinf(loss):
        print("error")
#         import pdb
#         pdb.set_trace()
        loss =0
  
    return loss


#ce loss
def _ce_loss(pred,gt):
   
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    NEAR_0 =1e-10
    
    alpha = 0.9
    
#     for pred in preds:
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = -alpha * torch.log(pos_pred + NEAR_0) * torch.pow(1 - pos_pred, 2)
    neg_loss = -(1-alpha) * torch.log(1 - neg_pred + NEAR_0)* torch.pow(neg_pred, 2) 

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    loss = (pos_loss + neg_loss)/gt.nelement()
  
    return loss


class CornerNet_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss,ce_loss=_ce_loss):
        super(CornerNet_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.off_loss    = _off_loss
        
        #added
        self.ce_loss = ce_loss

    #outs: output from forward; targets: groundtruths
    #outs: [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs,saliency_maps]
    def forward(self, outs, targets):
        #网络生成n_stacks个heats,tags,offs，不是只有一组
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_tags  = outs[2]
        br_tags  = outs[3]
        tl_offs  = outs[4]
        br_offs  = outs[5]
        class_preds = outs[6]
        
        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]
        gt_mask     = targets[2]
        gt_tl_off   = targets[3]
        gt_br_off   = targets[4]
        gt_tl_ind   = targets[5]
        gt_br_ind   = targets[6]
        gt_class_preds = targets[7]
        
        #class pred loss
        class_pred_loss = self.ce_loss(class_preds,gt_class_preds)
        class_pred_loss = class_pred_loss*5
        
        
        # focal loss
        focal_loss = 0
        
        f_n = len(tl_heats)
        
        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        #focal loss for heats
        focal_loss += self.focal_loss(tl_heats[0:f_n//2], gt_tl_heat[0])
        focal_loss += self.focal_loss(br_heats[0:f_n//2], gt_br_heat[0])
        
        
        #added by su
        focal_loss_d2 = 0
        focal_loss_d2 += self.focal_loss(tl_heats[f_n//2:f_n], gt_tl_heat[1])
        focal_loss_d2 += self.focal_loss(br_heats[f_n//2:f_n], gt_br_heat[1])
        
        
        focal_loss_d2 = focal_loss_d2 * 0.8

        # tag loss
        pull_loss = 0
        push_loss = 0
        
        #tmp var
        pull_loss_1=0
        push_loss_1=0
        
        #smooth_L1_loss for offset loss, t_pred - t_gt
        off_loss = 0
        off_loss_1 = 0
        
        scale_num = 2
        for i in range(scale_num):
 
            s_i = f_n*i//scale_num 
            e_i = f_n*(i+1)//scale_num
            
            #tag loss
            tl_tags_   = [_tranpose_and_gather_feat(tl_tag, gt_tl_ind[i]) for tl_tag in tl_tags[s_i:e_i]]
            br_tags_   = [_tranpose_and_gather_feat(br_tag, gt_br_ind[i]) for br_tag in br_tags[s_i:e_i]]
            for tl_tag, br_tag in zip(tl_tags_, br_tags_):
                pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
                pull_loss += pull
                push_loss += push
                if i==0:
                    pull_loss_1 += pull
                    push_loss_1 += push
                    
            
            #offset loss
            tl_offs_  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind[i]) for tl_off in tl_offs[s_i:e_i]]
            br_offs_  = [_tranpose_and_gather_feat(br_off, gt_br_ind[i]) for br_off in br_offs[s_i:e_i]]
            for tl_off, br_off in zip(tl_offs_, br_offs_):
                tmp_off_loss = self.off_loss(tl_off, gt_tl_off[i], gt_mask)
                tmp_off_loss += self.off_loss(br_off, gt_br_off[i], gt_mask)
                off_loss += tmp_off_loss
                if i==0:
                    off_loss_1 += tmp_off_loss
            
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss
        
        pull_loss_1 = self.pull_weight * pull_loss_1
        push_loss_1 = self.push_weight * push_loss_1

        off_loss = self.off_weight * off_loss
        
        
        print("loss-0:all={},class={}".format(focal_loss + focal_loss_d2 + pull_loss + push_loss + off_loss,class_pred_loss))
        print("loss-1:focal={},pull={},push={},off={}".format(focal_loss,pull_loss-pull_loss_1,push_loss-push_loss_1,off_loss-off_loss_1,))
        print("loss-2:focal={},pull={},push={},off={}".format(focal_loss_d2,pull_loss_1,push_loss_1,off_loss_1))
        
        #edited
        loss = (class_pred_loss+focal_loss + pull_loss + push_loss + off_loss + focal_loss_d2) / max(len(tl_heats), 1)
#         return loss.unsqueeze(0)
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), \
            (pull_loss / len(tl_heats)).unsqueeze(0),\
            (push_loss / len(tl_heats)).unsqueeze(0), \
            (off_loss / len(tl_heats)).unsqueeze(0)
