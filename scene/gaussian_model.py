#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2, getBoxes, customDist
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import torch.nn.functional as F
from tqdm import tqdm, trange

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def duplicate_pcd(self, pcd : BasicPointCloud):
        # Define mean and standard deviation for the normal distribution
        mean_points = pcd.points.mean()
        std_points = pcd.points.std()

        mean_normals = pcd.normals.mean()
        std_normals = pcd.normals.std()

        std_colors = pcd.colors.std()

        colors = np.repeat(pcd.colors, 3, axis=0)
        points = np.repeat(pcd.points, 3, axis=0)
        normals = np.repeat(pcd.normals, 3, axis=0)

        # # Iterate through the array
        # for i in range(int(points.shape[0] * 2 / 3)):
        #     # Sample 3 values from the normal distribution
        #     samples = np.random.normal(0, 1, 3)
        #
        #     # Add these samples to each element of the current row of the array
        #     points[i] += samples

        # # Iterate through the array
        # for i in range(int(normals.shape[0] * 2 / 3)):
        #     # Sample 3 values from the normal distribution
        #     samples = np.random.normal(0, std_normals, 3)
        #
        #     # Add these samples to each element of the current row of the array
        #     normals[i] += samples

        # # Iterate through the array
        # for i in range(int(colors.shape[0] * 2 / 3)):
        #     # Sample 3 values from the normal distribution
        #     samples = np.random.normal(0, std_colors, 3)
        #
        #     # Add these samples to each element of the current row of the array
        #     colors[i] += samples

        new_pcd = BasicPointCloud(points, colors, normals)
        return new_pcd

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # pcd = self.duplicate_pcd(pcd)
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0  # 아마도 코드 실수 (없어도 상관 없음: 그냥 내 생각엔 얘네가 features [N, 3, 1:] 이후에 0이 들어 간다는 걸 말하고 싶었나 봄

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # dist3 = torch.clamp_min(customDist(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def calc_similarity(self, similarity_threshold=0.99999):
        pass

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def tensor_to_pcd(self, xyz_parameter):
        xyz_array = xyz_parameter.detach().cpu().numpy()
        zeros_array = np.zeros(xyz_parameter.detach().cpu().shape)

        pcd = BasicPointCloud(xyz_array, zeros_array, zeros_array)

        return pcd
    
    def get_boxes(self, pcd: BasicPointCloud):
        box_data = getBoxes(torch.from_numpy(np.asarray(pcd.points)).float().cuda())
        return box_data
    
    def set_opacity(self, opacity):
        self._opacity = nn.Parameter(opacity.requires_grad_(True))

    def densify_and_merge(self, dst_thh, cov_thh):

        def intersection(x, y):

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            tmp0 = np.intersect1d(x, y)

            return torch.from_numpy(tmp0).cuda()

        g_xyz = self._xyz.clone()
        g_cov = self.covariance_activation(self.get_scaling, 1, self._rotation).clone()
        g_scaling = self.scaling_inverse_activation(self.get_scaling).clone()
        g_rotation = self._rotation.clone()
        g_opacity = self._opacity.clone()

        for i in tqdm(range(len(g_xyz)-1)):
            if torch.isnan(g_xyz[i][0]):
                continue

            tmp1 = torch.linalg.norm(g_xyz[i] - g_xyz[i + 1:], ord=2)
            indices1 = torch.where(tmp1 < dst_thh)[0]

            if indices1.shape[0] > 0:
                tmp2 = torch.linalg.norm(g_cov[i] - g_cov[i + 1:], ord=2)

                indices2 = torch.where(tmp2 < cov_thh)[0]

                merge_indices = intersection(indices1, indices2)

                # xyz, rotation, scaling, opacity을 mean으로 조정
                g_xyz[i] = torch.mean(g_xyz[merge_indices], dim=0)
                g_xyz[merge_indices] = float('nan')

                g_scaling[i] = torch.mean(g_scaling[merge_indices], dim=0)
                g_scaling[merge_indices] = float('nan')

                g_rotation[i] = torch.mean(g_rotation[merge_indices], dim=0)
                g_rotation[merge_indices] = float('nan')

                g_opacity[i] = torch.mean(g_opacity[merge_indices], dim=0)
                g_opacity[merge_indices] = float('nan')

        # new_g_xyz = g_xyz[~torch.isnan(g_xyz).any(dim=1)]
        # new_g_scale = self._scaling
        print('before merge')
        print(g_xyz.shape)

        self._xyz = g_xyz[~torch.isnan(g_xyz).any(dim=1)]
        self._scaling = g_scaling[~torch.isnan(g_xyz).any(dim=1)]
        self._rotation = g_rotation[~torch.isnan(g_xyz).any(dim=1)]
        self._features_dc = self._features_dc[~torch.isnan(g_xyz).any(dim=1)]
        self._features_rest = self._features_rest[~torch.isnan(g_xyz).any(dim=1)]
        self._opacity = g_opacity[~torch.isnan(g_xyz).any(dim=1)]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print('after merge')
        print(self._xyz.shape)

        # self.dens

    def flatten_gaussians(self):
        accum_scale_ratio = 0.0
        for i in range(len(self._xyz)):
            # scales = self._scaling[i].detach()
            # scales_min_idx = torch.argmin(scales)
            # scales = self._scaling.clone()
            scales_min_idx = torch.argmin(self.get_scaling[i])

            a = self._scaling[i][scales_min_idx]
            b = self.scaling_inverse_activation(
                self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])
            )
            a_scaled = self.get_scaling[i][scales_min_idx]
            b_scaled = self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])

            # self._scaling[i][scales_min_idx] = self.scaling_inverse_activation(
            #     self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])
            # )

            ## divided by 10
            # self._scaling[i][scales_min_idx] = self.scaling_inverse_activation(
            #     self.get_scaling[i][scales_min_idx] / 10.0
            # )

            self._scaling[i][scales_min_idx] = self.scaling_inverse_activation(
                ((self.get_scaling[i][scales_min_idx] * 1000.0) / (torch.sum(self.get_scaling[i]) * 1000.0)) / 1000.0
            )

        #     new_scale = self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])
        #
        #     # scaler = 2.0
        #     # new_scale = torch.minimum(torch.tensor(1.0),
        #     #                           (self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])) * scaler)
        #
        #     self._scaling[i][scales_min_idx] = self.scaling_inverse_activation(new_scale)
        #
        #     accum_scale_ratio += new_scale
        #
        # averaged_accum_scale = accum_scale_ratio / len(self._xyz)
        # print("\naverage scale ratio:", averaged_accum_scale)

            # self._scaling[i][scales_min_idx] = scales[i][scales_min_idx] / torch.sum(scales[i])

        # optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]
        #
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        print("Gaussians flattened.")
        print("memory:", torch.cuda.memory_allocated() / 1024 / 1024)

    def flatten_duplicate_gaussians(self):
        all_selected_mask = torch.ones(self.get_xyz.shape[0], device="cuda", dtype=bool)

        new_xyz = self._xyz[all_selected_mask]
        new_features_dc = self._features_dc[all_selected_mask]
        new_features_rest = self._features_rest[all_selected_mask]
        new_opacities = self._opacity[all_selected_mask]
        new_scaling = self._scaling[all_selected_mask]
        new_rotation = self._rotation[all_selected_mask]

        accum_scale_ratio = 0.0

        for i in range(len(self._xyz)):
            scales_min_idx = torch.argmin(self.get_scaling[i])

            # new_scale = self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])

            # scaler = 1.0
            # new_scale = torch.minimum(torch.tensor(1.0),
            #                           (self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])) * scaler)

            new_scale = ((self.get_scaling[i][scales_min_idx] * 1000.0) / (torch.sum(self.get_scaling[i]) * 1000.0)) / 1000.0

            new_scaling[i][scales_min_idx] = self.scaling_inverse_activation(
                new_scale
            )

            accum_scale_ratio += new_scale

        averaged_accum_scale = accum_scale_ratio / len(self._xyz)
        print("\naverage scale ratio:", averaged_accum_scale)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        print("Gaussians flattened.")


    # def flatten_duplicate_gaussians(self):
    #     all_selected_mask = torch.ones(self.get_xyz.shape[0], device="cuda", dtype=bool)
    #
    #     new_xyz = self._xyz[all_selected_mask]
    #     new_features_dc = self._features_dc[all_selected_mask]
    #     new_features_rest = self._features_rest[all_selected_mask]
    #     new_opacities = self._opacity[all_selected_mask]
    #     new_scaling = self._scaling[all_selected_mask]
    #     new_rotation = self._rotation[all_selected_mask]
    #
    #     for i in range(len(self._xyz)):
    #         scales_min_idx = torch.argmin(self.get_scaling[i])
    #
    #         new_scaling[i][scales_min_idx] = self.scaling_inverse_activation(
    #             self.get_scaling[i][scales_min_idx] / torch.sum(self.get_scaling[i])
    #         )
    #
    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    #
    #     print("\nGaussians flattened.")