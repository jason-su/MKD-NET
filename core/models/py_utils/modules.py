#coding:utf-8
import torch
import torch.nn as nn

from .utils import residual, upsample, merge, _decode, convolution
from .visualize import visualize,visualize_feature

def _make_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def _make_layer_revr(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)

def _make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def _make_unpool_layer(dim):
    return upsample(scale_factor=2)

def _make_merge_layer(dim):
    return merge()

#added
class pred_cls_module(nn.Module):
    def __init__(self,cls_n):
        super(pred_cls_module, self).__init__()
        self.cls_n = cls_n
        
        in_filters = 256
        
#         self.cn1 = convolution(1, 256, 256, with_bn=False)
        self.cn1 = nn.Sequential(
            convolution(1,in_filters, in_filters),
            convolution(3,in_filters,in_filters*2),
            convolution(1,in_filters*2,in_filters),
            convolution(3,in_filters,in_filters*2),
            convolution(1,in_filters*2,in_filters)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
#         self.cn2 = convolution(1,in_filters,in_filters//2)
#         self.cn3 = nn.Conv2d(in_filters//2, 1, (1, 1))
        self.fc1 = nn.Linear(256, self.cls_n)
        
#         self.fc2 = nn.Linear(1024,self.cls_n)
#         self.sigmod = torch.sigmoid()
        
    def forward(self, x):
        x_cn1 = self.cn1(x)
        
        x_gap1 = self.gap(x_cn1).view(x_cn1.size(0), -1)
#         x_cn2 = self.cn2(x_cn1)
#         x_cn3 = self.cn3(x_cn2)
        #batch_size*64*64*1 -> batch_size*4096
#         x_cn3 = x_cn3.view(x_cn3.size(0), -1)
        
        x_fc1 = self.fc1(x_gap1)
        output = torch.sigmoid(x_fc1)
        return output


class hg_module(nn.Module):
    def __init__(
        self, n, dims, modules, make_up_layer=_make_layer,
        make_pool_layer=_make_pool_layer, make_hg_layer=_make_layer,
        make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,
        make_unpool_layer=_make_unpool_layer, make_merge_layer=_make_merge_layer
    ):
        super(hg_module, self).__init__()
        
        #added by su
        self.n    = n
        self.up1s = nn.ModuleList() #equal list in python
        self.max1s = nn.ModuleList()
        self.low1s = nn.ModuleList()
        self.low2s = nn.ModuleList()
        self.low3s = nn.ModuleList()
        self.up2s = nn.ModuleList()
        self.mergs = nn.ModuleList()
         
        for i in range(n):
            curr_mod = modules[i]
            next_mod = modules[i+1]
     
            curr_dim = dims[i]
            next_dim = dims[i+1]
             
            self.up1s.append(make_up_layer(curr_dim, curr_dim, curr_mod))
            self.max1s.append(make_pool_layer(curr_dim))
            self.low1s.append(make_hg_layer(curr_dim, next_dim, curr_mod))  #squeeze: down/2
             
            self.low2s.append(make_low_layer(next_dim, next_dim, next_mod))
             
            self.low3s.append(make_hg_layer_revr(next_dim, curr_dim, curr_mod))
            self.up2s.append(make_unpool_layer(curr_dim)) #squeeze: up*2
            self.mergs.append(make_merge_layer(curr_dim))
        
    
        
        #4, [256, 256, 384, 384, 512], [2, 2, 2, 2, 4],
#         curr_mod = modules[0]
#         next_mod = modules[1]
#  
#         curr_dim = dims[0]
#         next_dim = dims[1]
#  
#         self.n    = n
#         self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
#         self.max1 = make_pool_layer(curr_dim)
#         self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)  #squeeze: down/2
#          
#         self.low2 = hg_module(
#             n - 1, dims[1:], modules[1:],
#             make_up_layer=make_up_layer,
#             make_pool_layer=make_pool_layer,
#             make_hg_layer=make_hg_layer, #squeeze: down/2
#             make_low_layer=make_low_layer,
#             make_hg_layer_revr=make_hg_layer_revr,
#             make_unpool_layer=make_unpool_layer,#squeeze: up*2
#             make_merge_layer=make_merge_layer
#         ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
#          
#         self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
#         self.up2  = make_unpool_layer(curr_dim) #squeeze: up*2
#         self.merg = make_merge_layer(curr_dim)
                

    def forward(self, x):
        up1 = []
        xs = []
        
        for i in range(self.n):
            up1.append(self.up1s[i](x))
            max1 = self.max1s[i](x)
            x = self.low1s[i](max1)
         
        x = self.low2s[self.n-1](x)
          
        for i in range(self.n-1,-1,-1):
            low3 = self.low3s[i](x)
            up2  = self.up2s[i](low3)
            x = self.mergs[i](up1[i], up2)
            xs.append(x)
         
        return xs[self.n-1],xs[self.n-2]
    
#         up1  = self.up1(x)
#         max1 = self.max1(x)
#         low1 = self.low1(max1)
#          
#         low2 = self.low2(low1)
#          
#         low3 = self.low3(low2)
#         up2  = self.up2(low3)
#         merg = self.merg(up1, up2)
#         return merg

class hg(nn.Module):
    def __init__(self, pre, hg_modules, cnvs, inters, cnvs_, inters_,subs,merges,cnv_d2s):
        super(hg, self).__init__()

        #预处理
        self.pre  = pre
        #hg模块 
        self.hgs  = hg_modules
        #n个cnvs
        self.cnvs = cnvs    #conv
        
        #n-1个下面的模块
        self.inters  = inters   #residual
        self.inters_ = inters_  #conv+bn
        self.cnvs_   = cnvs_    #conv+bn
        
        #added by su
        self.subs = subs
        self.merges = merges
        self.cnv_d2s = cnv_d2s
        

    def forward(self, x):
        inter = self.pre(x)

        cnvs  = []
        cnvs_d2 = []
        
        for ind, (hg_, cnv_) in enumerate(zip(self.hgs, self.cnvs)):
            hg,hg_d2  = hg_(inter)
            cnv = cnv_(hg)  #conv
            cnvs.append(cnv)
    
            
            #added by su
#             cnv_d2 = self.merges[ind](hg_d2) + self.subs[ind](hg)
            cnv_d2 = self.cnv_d2s[ind](hg_d2)
            cnvs_d2.append(cnv_d2)

            if ind < len(self.hgs) - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.functional.relu_(inter)
                inter = self.inters[ind](inter)
        
        cnvs.extend(cnvs_d2)
        return cnvs

#hour-glass 主干网结构
class hg_net(nn.Module):
    def __init__(
        self, hg, tl_modules, br_modules, tl_heats, br_heats, 
        tl_tags, br_tags, tl_offs, br_offs
    ):
        super(hg_net, self).__init__()

        self._decode = _decode

        self.hg = hg

        self.tl_modules = tl_modules
        self.br_modules = br_modules

        self.tl_heats = tl_heats
        self.br_heats = br_heats

        self.tl_tags = tl_tags
        self.br_tags = br_tags
        
        self.tl_offs = tl_offs
        self.br_offs = br_offs
        
        self.con1 = convolution(3, 1, 1, stride=1)
        self.sigm_ = nn.Sigmoid()
        
        self.test_id = 0
        
        self.pred_cls_op = pred_cls_module(20)

    def _train(self, xs):
        image =xs[0]
        cnvs  = self.hg(image)
        
        #添加一个分支；输入：cnvs
        #输出一个变量(batch_size,15)
        class_pred = self.pred_cls_op(cnvs[1])
        
        
        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        
        tl_heats_new = []
        br_heats_new = []
        
        #通道注意力乘法的写法2
        class_pred = class_pred.unsqueeze(2)
        
        for tl_heat_,br_heat_ in zip(tl_heats,br_heats):
            batch, cat, height, width = tl_heat_.size()
            tl_heat_ = tl_heat_.view(batch,cat,-1)
            br_heat_ = br_heat_.view(batch,cat,-1)
            
            #通道注意力乘法的写法2，速度较快
            #(batch, cat,-1) * (batch, cat,1)
            tl_heat_ = tl_heat_*class_pred
            br_heat_ = br_heat_*class_pred
            
            tl_heat_ = tl_heat_.view(batch, cat, height, width)
            br_heat_ = br_heat_.view(batch, cat, height, width)
            
            tl_heats_new.append(tl_heat_)
            br_heats_new.append(br_heat_)
        
        #通道注意力乘法的写法2
        class_pred = class_pred.squeeze(2)
        
#         print("tl_heat size:{}, saliency map size:".format(tl_heats[0].shape,saliency_maps[0].shape))
        
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        return [tl_heats_new, br_heats_new, tl_tags, br_tags, tl_offs, br_offs,class_pred]

    def _test(self, xs, **kwargs):
        image = xs[0]
        cnvs  = self.hg(image)
#         visualize_feature(saliency_map,image)
        
        class_pred = self.pred_cls_op(cnvs[1])
        
        multi_dets = []
        
        self.test_id +=1
#         for cnv_index in range(len(cnvs)):
        for cnv_index in [1,3]:
            tl_mod = self.tl_modules[cnv_index](cnvs[cnv_index])
            br_mod = self.br_modules[cnv_index](cnvs[cnv_index])
            
            
            tl_heat_o, br_heat_o = self.tl_heats[cnv_index](tl_mod), self.br_heats[cnv_index](br_mod)
            
            class_pred = class_pred.unsqueeze(2)
        
            batch, cat, height, width = tl_heat_o.size()
            tl_heat_o = tl_heat_o.view(batch,cat,-1)
            br_heat_o = br_heat_o.view(batch,cat,-1)
            
#      
            
            tl_heat = tl_heat_o*class_pred
            br_heat = br_heat_o*class_pred
            
            tl_heat = tl_heat.view(batch, cat, height, width)
            br_heat = br_heat.view(batch, cat, height, width)
            
            class_pred = class_pred.squeeze(2)
            
            tl_tag,  br_tag  = self.tl_tags[cnv_index](tl_mod),  self.br_tags[cnv_index](br_mod)
            tl_off,  br_off  = self.tl_offs[cnv_index](tl_mod),  self.br_offs[cnv_index](br_mod)
    
            outs = [tl_heat, br_heat, tl_tag, br_tag, tl_off, br_off]
            
            
#             hy = [[3,12,12,0.5],[3,25,25,0.5]]#1:41
            hy = [[3,20,20,0.5],[3,10,10,0.5]]#2:41.2
            hy = [[3,25,15,0.5],[3,10,10,0.5]]#3: 20-42.7
#             hy = [[3,20,20,0.5],[3,10,15,0.5]]#3: 41.8
            
            hy_i = 0
            if cnv_index >1:
                hy_i = 1
            
            kwargs["kernel"] = hy[hy_i][0]
            kwargs["num_dets"] = hy[hy_i][1]
            kwargs["K"] = hy[hy_i][2]
            kwargs["ae_threshold"] = hy[hy_i][3]
            
            multi_dets.append(self._decode(*outs, **kwargs))
         
        return multi_dets
    
        
    def forward(self, xs, test=False, **kwargs):
        if not test:
            return self._train(xs)
        return self._test(xs, **kwargs)
    
    