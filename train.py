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
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    repo = "isl-org/ZoeDepth"
    # Zoe_N
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)
    # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
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

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # # zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
        # import vidar.arch.networks.perceiver.ZeroDepthNet as ZeroDepthNet
        # import vidar.utils.config as config
        # # import scene.dataset_readers as dataset_readers
        # zerodepth_model = ZeroDepthNet.ZeroDepthNet(config.read_config('zerodepth_config.yaml'))
        #
        # # Specify the path to your text file
        # file_path = './download/tandt_db/train/sparse/0'
        #
        # # Open the file and read its contents
        # with open(file_path, 'r') as file:
        #     content = file.read()
        #
        # # Print the contents
        # print(content)
        #
        # intrinsics = torch.tensor(viewpoint_cam.R)
        # rgb = gt_image
        # depth = zerodepth_model.forward(rgb, intrinsics)


        # depth_pred = zerodepth_model(rgb, intrinsics)

        # Depth by ZoeDepth
        if iteration % 1000 == 0:
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

            depth_diff = abs(depth_numpy_gt - depth_numpy_pred)

            # Colorize output
            from zoedepth.utils.misc import colorize, colors

            colored_gt = colorize(depth_numpy_gt, cmap='Reds_r')
            colored_pred = colorize(depth_numpy_pred, cmap='Reds_r')
            colored_diff = colorize(depth_diff, cmap='Reds_r')

            # save colored output
            # Assuming 'tensor' is your (3, 545, 980) tensor
            # Transpose it to (Height, Width, Channels)
            tensor_permuted = gt_image.permute(1, 2, 0)  # Rearrange the tensor dimensions

            # Normalize or scale if necessary
            # For example, if your tensor has values from 0 to 1
            tensor_numpy = tensor_permuted.cpu().detach().numpy()
            tensor_scaled = (tensor_numpy * 255).astype(np.uint8)
            fpath_colored = "input.png"
            Image.fromarray(tensor_scaled).save(fpath_colored)

            fpath_colored = "output_colored_gt.png"
            Image.fromarray(colored_gt).save(fpath_colored)

            fpath_colored = "output_colored_pred.png"
            Image.fromarray(colored_pred).save(fpath_colored)

            fpath_colored = "output_diff.png"
            Image.fromarray(colored_diff).save(fpath_colored)

            # Load the four images
            image1 = Image.open('output_colored_gt.png')
            image2 = Image.open('output_colored_pred.png')
            image3 = Image.open('output_diff.png')
            image4 = Image.open('input.png')

            # # Assuming all images are the same size, get dimensions of one image
            # width, height = image1.size
            #
            # # Create titles for each image
            # titles = ["Title 1", "Title 2", "Title 3", "Title 4"]
            #
            # # Vertical space for titles
            # title_space = 40
            # font_size = 80  # Adjust the size as needed
            # font = ImageFont.load_default()
            #
            # # Create a new empty image with twice the width and height, add extra space for titles
            # new_im = Image.new('RGB', (width * 2, height * 2 + title_space * 2))
            #
            # # Create a drawing context
            # draw = ImageDraw.Draw(new_im)
            # font = ImageFont.load_default()  # Load a default font
            #
            # # Define a function to paste images and titles
            # def paste_image_and_title(image, title, position):
            #     x, y = position
            #     title_width, title_height = draw.textsize(title, font=font)
            #     title_x = x + (width - title_width) // 2
            #     new_im.paste(image, (x, y + title_space))  # Adjust vertical position for title
            #     draw.text((title_x, y), title, fill="white", font=font, font_size=font_size)
            #
            # # Paste images and titles
            # paste_image_and_title(image1, titles[0], (0, 0))
            # paste_image_and_title(image2, titles[1], (width, 0))
            # paste_image_and_title(image3, titles[2], (0, height + title_space))
            # paste_image_and_title(image4, titles[3], (width, height + title_space))

            # Assuming all images are the same size, get dimensions of one image
            width, height = image1.size

            # Create a new empty image with twice the width and height
            new_im = Image.new('RGB', (width * 2, height * 2))

            # Paste the images into the new image
            new_im.paste(image1, (0, 0))
            new_im.paste(image2, (width, 0))
            new_im.paste(image3, (0, height))
            new_im.paste(image4, (width, height))

            # Save the new image
            new_im.save('combined_image.png')

            pass

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
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

                if iteration % 1000 == 0:
                    gaussians.calc_similarity()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

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
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
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
