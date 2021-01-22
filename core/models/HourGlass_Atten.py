import torch
import torch.nn as nn

from .py_utils import TopPool, BottomPool, LeftPool, RightPool

from .py_utils.utils import convolution, residual, corner_pool
from .py_utils.losses import CornerNet_Loss
from .py_utils.modules import hg_module, hg, hg_net

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(inp_dim, out_dim, modules):
    layers  = [residual(inp_dim, out_dim, stride=2)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class model(nn.Module):
    def _pred_mod(self, dim):
        return nn.Sequential(
            convolution(3, 256, 256, with_bn=False),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

    def __init__(self):
        super(model, self).__init__()
        
        stacks  = 1
        pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(128, 256, stride=2)
        )
        hg_mods = nn.ModuleList([
            hg_module(
                5, [256, 256, 384, 384, 384, 512], [2, 2, 2, 2, 2, 4],
                make_pool_layer=make_pool_layer,
                make_hg_layer=make_hg_layer
            ) for _ in range(stacks)
        ])
        cnvs    = nn.ModuleList([convolution(3, 256, 256) for _ in range(stacks)])
        inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])

        hgs = hg(pre, hg_mods, cnvs, inters, cnvs_, inters_) 


        self.loss = CornerNet_Loss(pull_weight=1e-1, push_weight=1e-1)
    
    def _train(self, xs):
        image =xs[0]
        cnvs  = self.hg(image)
        
#         print("feature map size: ({},{})",cnvs[0].shape)
        
        saliency_maps = [self.pix_atten(cnv) for cnv in cnvs]
        
#         print("saliency map min={},mean={},max={}".format(saliency_maps[0].min(),saliency_maps[0].mean(),saliency_maps[0].max()))

        tl_modules = [tl_mod_(cnv) for tl_mod_, cnv in zip(self.tl_modules, cnvs)]
        br_modules = [br_mod_(cnv) for br_mod_, cnv in zip(self.br_modules, cnvs)]
        
        tl_heats   = [tl_heat_(tl_mod) for tl_heat_, tl_mod in zip(self.tl_heats, tl_modules)]
        br_heats   = [br_heat_(br_mod) for br_heat_, br_mod in zip(self.br_heats, br_modules)]
        
#         print("tl_heat size:{}, saliency map size:".format(tl_heats[0].shape,saliency_maps[0].shape))
        
        #added
        #heat = heat * weight
        tl_heats = [tl_heat_.mul(saliency_map_) for tl_heat_,saliency_map_ in zip(tl_heats,saliency_maps)]
        br_heats = [br_heat_.mul(saliency_map_) for br_heat_,saliency_map_ in zip(br_heats,saliency_maps)]
        
        
        tl_tags    = [tl_tag_(tl_mod)  for tl_tag_,  tl_mod in zip(self.tl_tags,  tl_modules)]
        br_tags    = [br_tag_(br_mod)  for br_tag_,  br_mod in zip(self.br_tags,  br_modules)]
        tl_offs    = [tl_off_(tl_mod)  for tl_off_,  tl_mod in zip(self.tl_offs,  tl_modules)]
        br_offs    = [br_off_(br_mod)  for br_off_,  br_mod in zip(self.br_offs,  br_modules)]
        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs,saliency_maps]

    def _test(self, xs, **kwargs):
        image = xs[0]
        cnvs  = self.hg(image)

       

    def forward(self, xs, test=False, **kwargs):
        if not test:
            return self._train(xs)
        return self._test(xs, **kwargs)