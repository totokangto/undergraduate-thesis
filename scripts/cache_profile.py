#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
project_directory = '..'
## Use insert but not append to make sure the python search the project directory first
sys.path.insert(0, os.path.abspath(project_directory))

import tinycudann as tcnn
import commentjson as ctjs
import torch
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch.nn.functional as F
from scene import Scene, GaussianModel
scene="flame_steak"
postfixs=['F_4']
ntc_conf_paths=['../configs/cache/cache_'+postfix+'.json' for postfix in postfixs]
pcd_path='../test/flame_steak_suite/flame_steak_init/point_cloud/iteration_15000/point_cloud.ply'
save_paths=['../ntc/flame_steak_ntc_params_'+postfix+'.pth' for postfix in postfixs]
ntcs=[]
gaussians = GaussianModel(1)
gaussians.load_ply(pcd_path)


# In[ ]:


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def fetchXYZ(path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    return torch.tensor(xyz, dtype=torch.float, device="cuda")

def get_xyz_bound(xyz, percentile=80):
    ## Hard-code the coordinate of the corners here!!
    return torch.tensor([-20, -15,   5]).cuda(), torch.tensor([15, 10, 23]).cuda()

def get_contracted_xyz(xyz):
    xyz_bound_min, xyz_bound_max = get_xyz_bound(xyz, 80)
    normalzied_xyz=(xyz-xyz_bound_min)/(xyz_bound_max-xyz_bound_min)
    return normalzied_xyz

@torch.compile
def quaternion_multiply(a, b):
    a_norm=nn.functional.normalize(a)
    b_norm=nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_loss(q1, q2):
    cos_theta = F.cosine_similarity(q1, q2, dim=1)
    cos_theta = torch.clamp(cos_theta, -1+1e-7, 1-1e-7)
    return 1-torch.pow(cos_theta, 2).mean()

def l1loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


# In[ ]:


from ntc import NeuralTransformationCache
for idx, ntc_conf_path in enumerate(ntc_conf_paths):
    with open(ntc_conf_path) as ntc_conf_file:
        ntc_conf = ctjs.load(ntc_conf_file)
    model=tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=ntc_conf["encoding"], network_config=ntc_conf["network"]).to(torch.device("cuda"))
    ntc=NeuralTransformationCache(model,torch.tensor([0.,0.,0.]).cuda(),torch.tensor([0.,0.,0.]).cuda())
    ntc.load_state_dict(torch.load(save_paths[idx]))
    ntcs.append(ntc)


# In[ ]:


import time
import torch
gaussians.ntc = ntc
gaussians._rotate_sh = True

# Since torch.compile is JIT compilation, we need to call the function once to trigger the compilation
gaussians.query_ntc_eval()

for idx, ntc in enumerate(ntcs):
    gaussians.ntc = ntc
    torch.cuda.synchronize()
    start = time.time()
    for i in range(300):
        gaussians.query_ntc_eval()
    torch.cuda.synchronize()
    end = time.time()
    print(f"Time: {((end - start) / 300.0):.5f}s for {postfixs[idx]} in {scene} scene.")

