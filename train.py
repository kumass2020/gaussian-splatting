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

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
from scene.cameras import Camera
import numpy as np
import math
from PIL import Image


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    def load_zoe_depth():
        repo = "isl-org/ZoeDepth"
        # Zoe_N
        model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        zoe = model_zoe_n.to(DEVICE)

        return zoe

    zoe = load_zoe_depth()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            print("")
            print(f"[iteration {iteration}] Number of Gaussians: {gaussians.get_xyz.shape[0]}")

            # ############# Box ##############
            # pcd = gaussians.tensor_to_pcd(gaussians.get_xyz)
            # box_data = gaussians.get_boxes(pcd)
            # box_minn_tensor = box_data[:, :3]  # All rows, first 3 columns
            # box_maxx_tensor = box_data[:, 3:6]  # All rows, columns 3 to 5
            # box_scale = box_data[:, 6:]  # All rows, last column
            #
            # def is_point_within_boundaries(points, min_boundaries, max_boundaries):
            #     # Move data to GPU
            #     points = points.cuda()
            #     min_boundaries = min_boundaries.cuda()
            #     max_boundaries = max_boundaries.cuda()
            #
            #     # Expand dimensions for broadcasting
            #     points_expanded = points.unsqueeze(1)  # Shape: [num_points, 1, 3]
            #
            #     # Check if points are within boundaries
            #     within_min = points_expanded >= min_boundaries  # Shape: [num_points, num_boundaries, 3]
            #     within_max = points_expanded <= max_boundaries  # Shape: [num_points, num_boundaries, 3]
            #
            #     # Both conditions must be true for all coordinates
            #     within_boundaries = torch.all(within_min & within_max, dim=2)  # Shape: [num_points, num_boundaries]
            #
            #     return within_boundaries
            #
            # # Checking which points are within which boundaries
            # points_within_boundaries = is_point_within_boundaries(gaussians.get_xyz, box_minn_tensor,
            #                                                       box_maxx_tensor)
            #
            # # Convert the boolean tensor to an integer tensor
            # points_within_boundaries_int = points_within_boundaries.int()
            #
            # # Now apply argmax
            # first_boundary_indices = torch.argmax(points_within_boundaries_int, dim=1)
            #
            # # For finding all boundary indices for each point
            # all_boundary_indices = [torch.nonzero(points_within_boundaries[i]).squeeze() for i in
            #                         range(points_within_boundaries.shape[0])]
            # ################################

        # ############## CSV save ##############
        # import csv
        #
        # # Define the CSV file name
        # csv_file_name = 'all_boundary_indices.csv'
        #
        # # Write the list of indices to the CSV file
        # with open(csv_file_name, 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for indices in all_boundary_indices:
        #         # Check if indices is a tensor with just one element
        #         if torch.numel(indices) == 1:
        #             # Write a single element tensor as a one-element list
        #             writer.writerow([indices.item()])
        #         else:
        #             # Convert tensor to list and write
        #             writer.writerow(indices.tolist())


        # ############# per 5 iterations #############
        # # Pick a random Camera
        # counter = iteration % 5
        # if iteration == 1 or iteration % 5 == 0:
        #     if not viewpoint_stack:
        #         viewpoint_stack = scene.getTrainCameras().copy()
        #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # else:
        #     pass
        #
        # # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True
        #
        # bg = torch.rand((3), device="cuda") if opt.random_background else background
        #
        # if iteration % 5 == 0:
        #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # ############################################

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # ################ Output res ################
        # def crop_tensor_to_four(image_tensor):
        #     """
        #     This function takes an image tensor of shape (3, height, width) and crops it into
        #     four smaller tensors each of shape (3, height/2, width/2).
        #     """
        #     # Assuming the image tensor is in the shape of (3, height, width)
        #     _, height, width = image_tensor.shape
        #
        #     # Calculating the midpoints
        #     mid_height, mid_width = height // 2, width // 2
        #
        #     # Cropping the image into four parts
        #     top_left = image_tensor[:, :mid_height, :mid_width]
        #     top_right = image_tensor[:, :mid_height, mid_width:]
        #     bottom_left = image_tensor[:, mid_height:, :mid_width]
        #     bottom_right = image_tensor[:, mid_height:, mid_width:]
        #
        #     return top_left, top_right, bottom_left, bottom_right
        #
        # def crop_image_to_four(image):
        #     """
        #     This function takes a PIL image and crops it into four smaller images.
        #     """
        #     width, height = image.size
        #
        #     # Calculating the midpoints
        #     mid_width, mid_height = width // 2, height // 2
        #
        #     # Defining the box coordinates for the four crops
        #     top_left_box = (0, 0, mid_width, mid_height)
        #     top_right_box = (mid_width, 0, width, mid_height)
        #     bottom_left_box = (0, mid_height, mid_width, height)
        #     bottom_right_box = (mid_width, mid_height, width, height)
        #
        #     # Cropping the image into four parts
        #     top_left = image.crop(top_left_box)
        #     top_right = image.crop(top_right_box)
        #     bottom_left = image.crop(bottom_left_box)
        #     bottom_right = image.crop(bottom_right_box)
        #
        #     return top_left, top_right, bottom_left, bottom_right
        #
        # # top_left_image, top_right_image, bottom_left_image, bottom_right_image = crop_tensor_to_four(fixed_image)
        #
        # if iteration % 5 != 0:
        #     sr_image_name = viewpoint_cam.image_name
        #     sr_gt_image = Image.open('download/tandt_db/tandt/train/super_resolution/' + sr_image_name + '.jpg')
        #     sr_gt_image = [crop_image_to_four(sr_gt_image)][0][counter-1]
        #     sr_gt_image = PILtoTorch(sr_gt_image, (viewpoint_cam.image_width, viewpoint_cam.image_height))
        #
        #     sr_gt_image = sr_gt_image[:3, ...]
        #     loaded_mask = None
        #
        #     if sr_gt_image.shape[1] == 4:
        #         loaded_mask = sr_gt_image[3:4, ...]
        #
        #     sr_gt_image = sr_gt_image.cuda()
        #
        #     viewpoint_cam_fixed = Camera(uid=viewpoint_cam.uid, colmap_id=viewpoint_cam.colmap_id, R=viewpoint_cam.R,
        #                                  T=viewpoint_cam.T, FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
        #                                  image=gt_image, gt_alpha_mask=loaded_mask, image_name=viewpoint_cam.image_name, scale=1.0,
        #                                  data_device="cuda")
        #
        #     viewpoint_cam_fixed.image_height = viewpoint_cam.image_height * 2
        #     viewpoint_cam_fixed.image_width = viewpoint_cam.image_width * 2
        #
        #     render_pkg = render(viewpoint_cam_fixed, gaussians, pipe, bg)
        #     fixed_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
        #         "viewspace_points"], \
        #         render_pkg["visibility_filter"], render_pkg["radii"]
        #
        #     fixed_image = [crop_tensor_to_four(fixed_image)][0][counter-1]
        # ######################################


        # ############# FoV #############
        # # Calculate local translation vector
        # tx = -math.tan(math.radians(viewpoint_cam.FoVx) / 2.0)
        # ty = math.tan(math.radians(viewpoint_cam.FoVy) / 2.0)
        # local_translation = np.array([tx, ty, 0])
        #
        # # Transform to global translation
        # global_translation = viewpoint_cam.R @ local_translation
        #
        # viewpoint_cam_fixed = Camera(uid=viewpoint_cam.uid, colmap_id=viewpoint_cam.colmap_id, R=viewpoint_cam.R,
        #                              T=viewpoint_cam.T, FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
        #                              image=gt_image, gt_alpha_mask=None, image_name=viewpoint_cam.image_name,
        #                              trans=global_translation, scale=1.0, data_device="cuda")
        #
        # viewpoint_cam_fixed.FoVx = viewpoint_cam_fixed.FoVx / 2.0
        # viewpoint_cam_fixed.FoVy = viewpoint_cam_fixed.FoVy / 2.0
        #
        # # Update camera center
        # viewpoint_cam_fixed.camera_center = viewpoint_cam_fixed.camera_center.cpu().detach().numpy() + global_translation
        # viewpoint_cam_fixed.camera_center = torch.from_numpy(viewpoint_cam_fixed.camera_center).float().to('cuda')
        #
        # viewpoint_cam_fixed.projection_matrix = getProjectionMatrix(znear=viewpoint_cam_fixed.znear,
        #                                                             zfar=viewpoint_cam_fixed.zfar,
        #                                                             fovX=viewpoint_cam_fixed.FoVx,
        #                                                             fovY=viewpoint_cam_fixed.FoVy).transpose(0,
        #                                                                                                      1).cuda()
        # viewpoint_cam_fixed.full_proj_transform = (viewpoint_cam_fixed.world_view_transform.unsqueeze(0).bmm(
        #     viewpoint_cam_fixed.projection_matrix.unsqueeze(0))).squeeze(0)
        # render_pkg = render(viewpoint_cam_fixed, gaussians, pipe, bg)
        # fixed_image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
        #     "viewspace_points"], \
        #     render_pkg["visibility_filter"], render_pkg["radii"]
        # ###############################

        def depth_loss():
            # Depth by ZoeDepth
            if iteration % 100 == 0:
                # Local file
                from PIL import Image, ImageDraw, ImageFont
                import torchvision.transforms as transforms
                import numpy as np

                _gt_image = viewpoint_cam.original_image.cpu()  # Move to CPU if it's on CUDA
                pil_image_gt = transforms.ToPILImage()(_gt_image.squeeze(0)).convert("RGB")
                depth_numpy_gt = zoe.infer_pil(pil_image_gt)  # as numpy
                depth_pil = zoe.infer_pil(pil_image_gt, output_type="pil")  # as 16-bit PIL Image
                depth_tensor = zoe.infer_pil(pil_image_gt, output_type="tensor")  # as torch tensor

                pred_image = image.cpu()  # Move to CPU if it's on CUDA
                pil_image_pred = transforms.ToPILImage()(pred_image.squeeze(0)).convert("RGB")
                depth_numpy_pred = zoe.infer_pil(pil_image_pred)

                depth_torch_gt = torch.from_numpy(depth_numpy_gt)
                depth_torch_pred = torch.from_numpy(depth_numpy_pred)

                # 두 텐서 간의 차이 계산
                # depth_diff_torch = abs(depth_torch_gt - depth_torch_pred)
                depth_diff_torch = depth_torch_gt - depth_torch_pred

                depth_diff_torch = depth_diff_torch.to("cuda")
                depth_diff_torch = depth_diff_torch.unsqueeze(0)
                depth_diff_torch = depth_diff_torch.mean()

                # # Colorize output
                # from zoedepth.utils.misc import colorize, colors
                #
                # colored_gt = colorize(depth_numpy_gt, cmap='Reds_r')
                # colored_pred = colorize(depth_numpy_pred, cmap='Reds_r')
                # colored_diff = colorize(depth_diff, cmap='Reds_r')
                #
                # # save colored output
                # # Assuming 'tensor' is your (3, 545, 980) tensor
                # # Transpose it to (Height, Width, Channels)
                # tensor_permuted = gt_image.permute(1, 2, 0)  # Rearrange the tensor dimensions
                #
                # # Normalize or scale if necessary
                # # For example, if your tensor has values from 0 to 1
                # tensor_numpy = tensor_permuted.cpu().detach().numpy()
                # tensor_scaled = (tensor_numpy * 255).astype(np.uint8)
                # fpath_colored = "input.png"
                # Image.fromarray(tensor_scaled).save(fpath_colored)
                #
                # fpath_colored = "output_colored_gt.png"
                # Image.fromarray(colored_gt).save(fpath_colored)
                #
                # fpath_colored = "output_colored_pred.png"
                # Image.fromarray(colored_pred).save(fpath_colored)
                #
                # fpath_colored = "output_diff.png"
                # Image.fromarray(colored_diff).save(fpath_colored)
                #
                # # Load the four images
                # image1 = Image.open('output_colored_gt.png')
                # image2 = Image.open('output_colored_pred.png')
                # image3 = Image.open('output_diff.png')
                # image4 = Image.open('input.png')
                #
                # # # Assuming all images are the same size, get dimensions of one image
                # # width, height = image1.size
                # #
                # # # Create titles for each image
                # # titles = ["Title 1", "Title 2", "Title 3", "Title 4"]
                # #
                # # # Vertical space for titles
                # # title_space = 40
                # # font_size = 80  # Adjust the size as needed
                # # font = ImageFont.load_default()
                # #
                # # # Create a new empty image with twice the width and height, add extra space for titles
                # # new_im = Image.new('RGB', (width * 2, height * 2 + title_space * 2))
                # #
                # # # Create a drawing context
                # # draw = ImageDraw.Draw(new_im)
                # # font = ImageFont.load_default()  # Load a default font
                # #
                # # # Define a function to paste images and titles
                # # def paste_image_and_title(image, title, position):
                # #     x, y = position
                # #     title_width, title_height = draw.textsize(title, font=font)
                # #     title_x = x + (width - title_width) // 2
                # #     new_im.paste(image, (x, y + title_space))  # Adjust vertical position for title
                # #     draw.text((title_x, y), title, fill="white", font=font, font_size=font_size)
                # #
                # # # Paste images and titles
                # # paste_image_and_title(image1, titles[0], (0, 0))
                # # paste_image_and_title(image2, titles[1], (width, 0))
                # # paste_image_and_title(image3, titles[2], (0, height + title_space))
                # # paste_image_and_title(image4, titles[3], (width, height + title_space))
                #
                # # Assuming all images are the same size, get dimensions of one image
                # width, height = image1.size
                #
                # # Create a new empty image with twice the width and height
                # new_im = Image.new('RGB', (width * 2, height * 2))
                #
                # # Paste the images into the new image
                # new_im.paste(image1, (0, 0))
                # new_im.paste(image2, (width, 0))
                # new_im.paste(image3, (0, height))
                # new_im.paste(image4, (width, height))
                #
                # # Save the new image
                # new_im.save('combined_image.png')

                if iteration % 100 == 0:
                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + depth_diff_torch
            else:
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            return loss

        # ############## SR loss ##############
        # if iteration % 5 != 0:
        #     Ll1 = l1_loss(fixed_image, sr_gt_image)
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(fixed_image, sr_gt_image))
        # else:
        #     Ll1 = l1_loss(image, gt_image)
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        ############## vanilla loss ##############
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # depth loss
        loss = depth_loss()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # if iteration % 1000 == 0:
                #     gaussians.calc_similarity()


                def get_box_boundary():
                    ################## box boundary #########################
                    if iteration % 1000 == 0 and iteration < 15000:
                        pcd = gaussians.tensor_to_pcd(gaussians.get_xyz)
                        box_data = gaussians.get_boxes(pcd)
                        box_minn_tensor = box_data[:, :3]  # All rows, first 3 columns
                        box_maxx_tensor = box_data[:, 3:6]  # All rows, columns 3 to 5
                        box_scale = box_data[:, 6:]  # All rows, last column

                        def is_point_within_boundaries(points, min_boundaries, max_boundaries):
                            # Move data to GPU
                            points = points.cuda()
                            min_boundaries = min_boundaries.cuda()
                            max_boundaries = max_boundaries.cuda()

                            # Expand dimensions for broadcasting
                            points_expanded = points.unsqueeze(1)  # Shape: [num_points, 1, 3]

                            # Check if points are within boundaries
                            within_min = points_expanded >= min_boundaries  # Shape: [num_points, num_boundaries, 3]
                            within_max = points_expanded <= max_boundaries  # Shape: [num_points, num_boundaries, 3]

                            # Both conditions must be true for all coordinates
                            within_boundaries = torch.all(within_min & within_max, dim=2)  # Shape: [num_points, num_boundaries]

                            return within_boundaries

                        # Checking which points are within which boundaries
                        points_within_boundaries = is_point_within_boundaries(gaussians.get_xyz, box_minn_tensor,
                                                                              box_maxx_tensor)

                        # Convert the boolean tensor to an integer tensor
                        points_within_boundaries_int = points_within_boundaries.int()

                        # Now apply argmax
                        first_boundary_indices = torch.argmax(points_within_boundaries_int, dim=1)

                        # For finding all boundary indices for each point
                        all_boundary_indices = [torch.nonzero(points_within_boundaries[i]).squeeze() for i in
                                                range(points_within_boundaries.shape[0])]

                        return all_boundary_indices
                    ######################################################
                # all_boundary_indices = get_box_boundary()

                def prune_points_most_boxes(all_boundary_indices):
                    ############## points pruning with most boxes ##############
                    # Step 1: Mask for elements with only {bounding_box_num} boundary
                    bounding_box_num = max(
                        (sub_tensor.size(0) for sub_tensor in all_boundary_indices if sub_tensor.dim() > 0), default=0)
                    # bounding_box_num = 0
                    one_boundary_mask = torch.tensor(
                        [indices.numel() == bounding_box_num for indices in all_boundary_indices])

                    print(f"\n[iteration {iteration}] Number of Gaussians before pruning: {gaussians.get_xyz.shape[0]}")
                    gaussians.prune_points(one_boundary_mask)
                    print(f"[iteration {iteration}] Number of Gaussians after pruning: {gaussians.get_xyz.shape[0]}")
                    ############################################################
                # prune_points_most_boxes(all_boundary_indices)

                def box_indices_to_csv(all_boundary_indices):
                    ############## CSV save ##############
                    import csv

                    # Define the CSV file name
                    csv_file_name = 'all_boundary_indices.csv'

                    # Write the list of indices to the CSV file
                    with open(csv_file_name, 'w', newline='') as file:
                        writer = csv.writer(file)
                        for indices in all_boundary_indices:
                            # Check if indices is a tensor with just one element
                            if torch.numel(indices) == 1:
                                # Write a single element tensor as a one-element list
                                writer.writerow([indices.item()])
                            else:
                                # Convert tensor to list and write
                                writer.writerow(indices.tolist())
                # box_indices_to_csv(all_boundary_indices)

                    # ############## box mask ##############
                    # # Assuming all_boundary_indices is a list of tensors
                    # # Reshape each tensor in the list to be 1D if it's not already
                    # reshaped_all_boundary_indices = [indices.reshape(-1) for indices in all_boundary_indices]
                    #
                    # # Now concatenate the reshaped tensors
                    # flat_indices = torch.cat(reshaped_all_boundary_indices)
                    #
                    # # Create a mask where each element is True if it's equal to 3
                    # mask = (flat_indices == 3)
                    #
                    # # Count the number of True values in the mask
                    # count_boundary_index_3 = torch.sum(mask).item()
                    #
                    # # Initialize a variable to store the minimum length
                    # min_length = float('inf')  # Start with infinity
                    #
                    # # Iterate through all tensors in all_boundary_indices
                    # for indices in all_boundary_indices:
                    #     # Update min_length if the current tensor is smaller
                    #     min_length = min(min_length, indices.numel())
                    #
                    # # Create a list of the lengths of each tensor in all_boundary_indices
                    # lengths = [indices.numel() for indices in all_boundary_indices]
                    #
                    # # Convert the list of lengths to a PyTorch tensor
                    # lengths_tensor = torch.tensor(lengths).float().cuda()  # Ensure it's a floating point tensor
                    # #####################################

                    # ############## opacity ##############
                    # # Normalize to [0, 1]
                    # min_val = torch.min(lengths_tensor).cuda()
                    # max_val = torch.max(lengths_tensor).cuda()
                    # lengths_norm_tensor = (lengths_tensor - min_val) / (max_val - min_val)
                    #
                    # # Shift to [-1, 1]
                    # lengths_norm_tensor = (lengths_norm_tensor * 2 - 1) * 0.1
                    #
                    # gaussians.set_opacity(gaussians.get_opacity + lengths_norm_tensor.unsqueeze(0).T)
                    # #####################################

                    # ########## opacity - Beta ##########
                    # from torch.distributions.beta import Beta
                    # # Assuming lengths_tensor is already defined and is a tensor
                    # min_val = torch.min(lengths_tensor).cuda()
                    # max_val = torch.max(lengths_tensor).cuda()
                    # lengths_norm_tensor = (lengths_tensor - min_val) / (max_val - min_val)
                    #
                    # # Define alpha and beta parameters for the beta distribution
                    # alpha, beta = 0.5, 0.5  # Example values, change them according to your needs
                    # beta_distribution = Beta(alpha, beta)
                    #
                    # # Sample from the beta distribution
                    # beta_samples = beta_distribution.sample(lengths_norm_tensor.shape).cuda()
                    #
                    # # Now scale these samples to [-1, 1] to match the scaling of lengths_norm_tensor
                    # beta_samples_scaled = (beta_samples * 2 - 1) * 0.1
                    #
                    # print("beta")
                    #
                    # # Modify the opacity
                    # gaussians.set_opacity(gaussians.get_opacity + beta_samples_scaled.unsqueeze(0).T)
                    # #####################################

                    # ############## prune ##############
                    # box_scale = box_scale.cuda()
                    # min_val = torch.min(box_scale).cuda()
                    # max_val = torch.max(box_scale).cuda()
                    #
                    # box_scale_norm_tensor = (box_scale - min_val) / (max_val - min_val)
                    #
                    # mean = torch.mean(box_scale_norm_tensor)
                    # sigma = torch.std(box_scale_norm_tensor)
                    #
                    # # Create a boolean mask where the condition is true
                    # condition = box_scale_norm_tensor > mean + 2 * sigma
                    #
                    # # Get indices of elements satisfying the condition
                    # selected_indices = torch.nonzero(condition, as_tuple=False)[:, 0]
                    #
                    # # Step 1: Mask for elements with only one boundary
                    # bounding_box_num = 1
                    # one_boundary_mask = torch.tensor([indices.numel() <= bounding_box_num for indices in all_boundary_indices])
                    #
                    # # Step 2: Create a mask for each point in all_boundary_indices
                    # selected_indices_mask_per_point = [torch.any(torch.isin(indices, selected_indices)) for indices in
                    #                                    all_boundary_indices]
                    #
                    # # Convert the list of booleans to a tensor if needed
                    # selected_indices_mask_per_point_tensor = torch.tensor(selected_indices_mask_per_point)
                    #
                    # # Step 3: Combine the masks
                    # # Since flat_all_boundary_indices is flat, we need to reshape selected_indices_mask to match the shape of one_boundary_mask
                    # selected_indices_mask_reshaped = selected_indices_mask_per_point_tensor.view(len(all_boundary_indices), -1).any(
                    #     dim=1)
                    # final_mask = one_boundary_mask & selected_indices_mask_reshaped
                    # print(f"\n[iteration {iteration}] Number of Gaussians before pruning: {gaussians.get_xyz.shape[0]}")
                    # gaussians.prune_points(final_mask)
                    # print(f"[iteration {iteration}] Number of Gaussians after pruning: {gaussians.get_xyz.shape[0]}")


                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                ################## Merge ##################
                # Densification
                if iteration % 300 == 0 and 5000 <= iteration <= 12000:
                    print('before merge')
                    print(gaussians.get_xyz.shape[0])

                    dst_thh = sum((torch.max(gaussians.get_xyz, dim=0)[0] - torch.min(gaussians.get_xyz, dim=0)[0])) / \
                              gaussians.get_xyz.shape[0] / gaussians.get_xyz.shape[1]
                    cov_thh = sum((torch.max(gaussians.get_covariance(), dim=0)[0] -
                                   torch.min(gaussians.get_covariance(), dim=0)[0])) / gaussians.get_covariance().shape[
                                  0] / gaussians.get_covariance().shape[1]

                    gaussians.densify_and_merge(dst_thh, cov_thh)

                    print('after merge')
                    print(gaussians.get_xyz.shape[0])
                ###########################################


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--augment_points", type=bool, default = False)
    # parser.add_argument("--depth_loss", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
