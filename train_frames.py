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
import time
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, quaternion_loss, d_xyz_gt, d_rot_gt
from gaussian_renderer import render, network_gui
import sys
import json
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, extract_valid_motion_pixels, visualize_optical_flow
from utils.debug_utils import save_tensor_img
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import re
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training_one_frame(dataset, opt, pipe, load_iteration, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, frame_num, output_path):
    start_time=time.time()
    last_s1_res = []
    #last_s1h_res = []
    last_s2_res = []
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree,opt.rotate_sh)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)
    
    gaussians.training_one_frame_setup(opt, frame_num, output_path) # generate NTC instance and initialize MLP
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
    s1_start_time=time.time()
    # Train the NTC
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # iter가 150이라 실제론 사용 x, 3DGS에서 그냥 가져온 코드
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
                     
        # Query the NTC
        gaussians.query_ntc() # train NTC
        
        loss = torch.tensor(0.).cuda()
        
        
        # A simple 
        for batch_iteraion in range(opt.batch_size): # 1장
        
            # Pick a random Camera
            # 21장의 dataset 이미지 중에서 랜덤으로 1장 골라서 비교
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            Lds = torch.tensor(0.).cuda()
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
        loss/=opt.batch_size
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
            s1_res = training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s1_res is not None:
                last_s1_res.append(s1_res)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration=iteration, save_type='all')
            
            # Tracking Densification Stats
            if iteration > opt.densify_from_iter: # default : 130
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                viewspace_point_tensor_grad = viewspace_point_tensor.grad
                # get view-space gradient : 약 20만개
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.ntc_optimizer.step()
                gaussians.ntc_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations): # 저장 x
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.output_path + "/chkpnt" + str(iteration) + ".pth")

    s1_end_time=time.time()
    # Densify
    if(opt.iterations_s2>0):
    # Dump the NTC
        scene.dump_NTC()
    # Update Gaussians by NTC
        gaussians.update_by_ntc()
        
    # Prune, Clone and setting up  
        gaussians.training_one_frame_s3_setup(opt)
        progress_bar = tqdm(range(opt.iterations, opt.iterations + opt.iterations_s2), desc="Training progress of Stage 2")    

    # Train the new Gaussians
    for iteration in range(opt.iterations + 1, opt.iterations + opt.iterations_s2 + 1):            
        iter_start.record()
                     
        # Update Learning Rate
        # gaussians.update_learning_rate(iteration)
        
        loss = torch.tensor(0.).cuda()
        
        for batch_iteraion in range(opt.batch_size): # 1장
        
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss += (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            
        loss/=opt.batch_size
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (iteration - opt.iterations) % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations + opt.iterations_s2:
                progress_bar.close()

            # Log and save
            s2_res = training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if s2_res is not None:
                last_s2_res.append(s2_res)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                #scene.save(iteration=iteration, save_type='added')
                scene.save(iteration=iteration, save_type='all')
                      
            # Densification
            if (iteration - opt.iterations) % opt.densification_interval == 0: # every 20 iters
                gaussians.adding_and_prune(opt,scene.cameras_extent)
                    
            # Optimizer step
            if iteration < opt.iterations + opt.iterations_s2:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    s2_end_time=time.time()
    
    # Calculate total training time
    pre_time = s1_start_time - start_time
    s1_time = s1_end_time - s1_start_time
    s2_time = s2_end_time - s1_end_time
           
    return last_s1_res, last_s2_res, pre_time, s1_time, s2_time

def prepare_output_and_logger(args):    
    if not args.output_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.output_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.output_path))
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.output_path)
    else:
        print("Tensorboard not available: not logging progress")
    tb_writer = None
    return tb_writer


def training_report(tb_writer, iteration, Ll1, Lds, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    last_test_psnr=0.0

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ds_loss', Lds.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations: # only test at 150 and 250 iters
    #if iteration in range(0,150,10):    
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_of = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    # if scene.gaussians._added_mask is not None:
                    #     added_pkg = renderFunc(viewpoint, scene.gaussians.get_masked_gaussian(scene.gaussians._added_mask), *renderArgs)
                    image, depth = torch.clamp(render_pkg["render"], 0.0, 1.0), render_pkg["depth"]
                    depth_vis=depth/(depth.max()+1e-5)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/diff".format(viewpoint.image_name), (gt_image-image).abs().mean(dim=0, keepdim=True), global_step=iteration)
                        # tb_writer.add_image(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_vis, global_step=iteration)
                        # if scene.gaussians._added_mask is not None:
                        #     tb_writer.add_image(config['name'] + "_view_{}/added_gaussians".format(viewpoint.image_name), torch.clamp(added_pkg["render"], 0.0, 1.0), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])    
                psnr_of /= len(config['cameras'])     
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if config['name'] == 'test':
                    last_test_psnr = psnr_test
                    last_test_image = image
                    last_gt = gt_image

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
        return {'last_test_psnr':last_test_psnr.cpu().numpy()
                , 'last_test_image':last_test_image.cpu()
                , 'last_points_num':scene.gaussians.get_xyz.shape[0]
                # , 'last_gt':last_gt.cpu()
                }

def train_one_frame(lp,op,pp,args, frame_num, output_path):
    args.save_iterations.append(args.iterations + args.iterations_s2) # arguments/__init__.py 30,000 + 0    
    if args.depth_smooth==0:
        args.bwd_depth=False
    print("Optimizing " + args.output_path)
    res_dict={}
    if(args.opt_type=='3DGStream'):
        s1_ress, s2_ress, pre_time, s1_time, s2_time = training_one_frame(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, frame_num, output_path)
        # All done
        print("\nTraining complete.")
        print(f"Preparation: {pre_time}")
        if pre_time > 2:
            print(f"If preparation is time-consuming, consider down-scaling the images BEFORE running 3DGStream.")
        print(f"Stage 1: {s1_time}")
        print(f"Stage 2: {s2_time}")
        if s1_ress !=[]:
            for idx, s1_res in enumerate(s1_ress):
                if frame_num % 30 == 0:
                    save_tensor_img(s1_res['last_test_image'],os.path.join(args.output_path,f'f{frame_num}_rendering1'))
                res_dict[f'stage1/psnr']=s1_res['last_test_psnr']
                res_dict[f'stage1/points_num']=s1_res['last_points_num']
            res_dict[f'stage1/time']=s1_time
        if s2_ress !=[]:
            for idx, s2_res in enumerate(s2_ress):
                if frame_num % 30 == 0:
                    save_tensor_img(s2_res['last_test_image'],os.path.join(args.output_path,f'f{frame_num}_rendering2'))
                res_dict[f'stage2/psnr']=s2_res['last_test_psnr']
                res_dict[f'stage2/points_num']=s2_res['last_points_num']
            res_dict[f'stage2/time']=s2_time
    return res_dict

def train_frames(lp, op, pp, args):
    # Initialize system state (RNG)
    sum_dict = {
        'stage1/psnr' : 0.0,
        'stage2/psnr' : 0.0,
        'stage1/time' : 0.0,
        'stage2/time' : 0.0,
        'stage1/points_num' : 0,
        'stage2/points_num' : 0
    }

    frame_num = 0
    avg_num = 0
    time_avg = 0
    safe_state(args.quiet)
    video_path=args.video_path
    output_path=args.output_path
    model_path=args.model_path
    load_iteration = args.load_iteration
    sub_paths = os.listdir(video_path)
    pattern = re.compile(r'frame(\d+)')
    frames = sorted(
        (item for item in sub_paths if pattern.match(item)),
        key=lambda x: int(pattern.match(x).group(1))
    )
    
    frames=frames[args.frame_start:args.frame_end]
    
    if args.frame_start==1:
        args.load_iteration = args.first_load_iteration # 15000
        
    for frame in frames:
        frame_num += 1
        avg_num += 1
        
        start_time = time.time()
        args.source_path = os.path.join(video_path, frame)
        args.output_path = os.path.join(output_path, frame)
        args.model_path = model_path
        frame_dict = train_one_frame(lp,op,pp,args, frame_num, output_path)
        
        time_avg += time.time()-start_time
        print(f"\nFrame {frame} finished in {time.time()-start_time} seconds.\n\n")
        
        sum_dict = {key: sum_dict[key] + frame_dict[key] for key in frame_dict.keys()}
        for name, value in sum_dict.items():
            print(f"\n {name} : {value/avg_num:.2f}")
        
        model_path = args.output_path
        args.load_iteration = load_iteration
        torch.cuda.empty_cache()
        
    print(f"\n Average training time is {time_avg/avg_num:.2f} seconds.\n\n")    
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--load_iteration', type=int, default=None) 
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.output_path == "":
        args.output_path=args.model_path
    if args.read_config and args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items(): # in test/flame_steak_suite/cfg_args.json, iterations is 150, iterations_s2 is 100
            if key not in ["output_path", "source_path", "model_path", "video_path", "debug_from"]:
                setattr(args, key, value)
    serializable_namespace = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    json_namespace = json.dumps(serializable_namespace)
    os.makedirs(args.output_path, exist_ok = True)
    with open(os.path.join(args.output_path, "cfg_args.json"), 'w') as f:
        f.write(json_namespace)

    # parameters : modelParams, optimizationParams, pipelineParams
    train_frames(lp,op,pp,args)
    """
    extent = 0 sh_degree = 1 ply_name = points3D.ply images = images_2 resolution = 1 white_background = False data_device = cuda

    eval = True iterations = 150 iterations_s2 = 100 first_load_iteration = 15000 position_lr_init = 0.0024 position_lr_final = 2.4e-05

    position_lr_delay_mult = 0.01 position_lr_max_steps = 30000 feature_lr = 0.0375 opacity_lr = 0.75 scaling_lr = 0.075

    rotation_lr = 0.015 percent_dense = 0.01 lambda_dssim = 0.2 depth_smooth = 0.0 ntc_lr = 0.002 lambda_dxyz = 0 lambda_drot = 0

    densification_interval = 20 opacity_reset_interval = 3000 densify_from_iter = 130 densify_until_iter = 15000 densify_grad_threshold = 0.00015

    ntc_conf_path = configs/cache/cache_F_4.json ntc_path = ntc/flame_steak_ntc_params_F_4.pth batch_size = 1 spawn_type = spawn

    s2_type = spawn s2_adding = True num_of_split = 1 num_of_spawn = 1 std_scale = 2 min_opacity = 0.01 rotate_sh = False only_mlp = False

    convert_SHs_python = False compute_cov3D_python = False debug = False bwd_depth = False opt_type = 3DGStream ip = 127.0.0.1

    port = 6009 detect_anomaly = False test_iterations = [150, 250] save_iterations = [150] frame_start = 1 frame_end = 300

    quiet = False checkpoint_iterations = [] start_checkpoint = None read_config = True load_iteration = 150
    """
