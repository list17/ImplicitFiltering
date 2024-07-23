# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.fields import Network
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
import mcubes
import warnings
import os
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        
        if args.cls is not None:
            self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], args.cls, args.dir)
        else:
            self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], args.dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname, args.cls)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.lambda_cd = args.lambda_cd
        
        self.ChamferDisL1 = ChamferDistanceL1().cuda()
        self.ChamferDisL2 = ChamferDistanceL2().cuda()

        self.mode = mode

        # Networks
        self.sdf_network = Network(**self.conf['model.sdf_network']).to(self.device)
        # self.grad_network = Network(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
        
        # self.load_checkpoint('ckpt_040000.pth')

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter

        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(self.iter_step)

            near_points, samples, point_gt = self.dataset_np.np_train_data(batch_size)
            B = near_points.shape[0]
            samples.requires_grad = True
            gradients_sample, sdf_sample = self.sdf_network.gradient_with_sdf(samples)
            grad_norm = F.normalize(gradients_sample, dim=1)               
            sample_moved = samples - grad_norm * sdf_sample                
            
            gradient_sample_moved, sdf_sample_moved = self.sdf_network.gradient_with_sdf(sample_moved)
            gradient_sample_moved_norm = F.normalize(gradient_sample_moved, dim=1)   
            
            near_points = near_points.reshape(-1, 3)
            gradient_near_points, sdf_near_points = self.sdf_network.gradient_with_sdf(near_points)
            gradient_near_points_norm = F.normalize(gradient_near_points, dim=-1)
            near_points = near_points.reshape(B, -1, 3)
            gradient_near_points_norm = gradient_near_points_norm.reshape(B, -1, 3)
            sdf_near_points = sdf_near_points.reshape(B, -1, 1)
    
            level_set_points = near_points + gradient_near_points_norm * sdf_sample.unsqueeze(-1)
            level_set_points = level_set_points.reshape(-1, 3)
            gradient_level_set_points, sdf_level_set_points = self.sdf_network.gradient_with_sdf(level_set_points)
            gradient_level_set_points_norm = F.normalize(gradient_level_set_points, dim=-1)
            level_set_points = level_set_points.reshape(B, -1, 3)
            gradient_level_set_points_norm = gradient_level_set_points_norm.reshape(B, -1, 3)
            sdf_level_set_points = sdf_level_set_points.reshape(B, -1, 1)
            
            zero_distances = ((sample_moved.unsqueeze(1) - near_points)**2).sum(-1)
            theta_r = torch.exp(-(zero_distances / torch.max(zero_distances, dim=-1, keepdim=True)[0]))
            phi_n = torch.exp(-((1 - (gradient_sample_moved_norm.unsqueeze(1) * gradient_near_points_norm).sum(dim=-1))/ (1-np.cos(15/180 * np.pi)))**2)
            
            weight = theta_r * phi_n + 1e-12
            weight = weight / weight.sum(-1, keepdim=True)
            project_dis = torch.abs(((sample_moved.unsqueeze(1) - near_points) * gradient_near_points_norm).sum(-1))
            project_dis_inv = torch.abs(((sample_moved.unsqueeze(1) - near_points) * gradient_sample_moved_norm.unsqueeze(1)).sum(-1))
            
            loss_filter_zero = ((project_dis + project_dis_inv) * weight).sum(-1).mean()
            

            level_distances = ((samples.unsqueeze(1) - level_set_points)**2).sum(-1)
            theta_r_level = torch.exp(-(level_distances / torch.max(level_distances, dim=-1, keepdim=True)[0]))
            phi_n_level = torch.exp(-((1 - (grad_norm.unsqueeze(1) * gradient_level_set_points_norm).sum(dim=-1))/ (1-np.cos(15/180 * np.pi)))**2)
            
            weight_level = theta_r_level * phi_n_level + 1e-12
            weight_level = weight_level / weight_level.sum(-1, keepdim=True)
            project_dis_level = torch.abs(((samples.unsqueeze(1) - level_set_points) * gradient_level_set_points_norm).sum(-1))
            project_dis2_level = torch.abs(((samples.unsqueeze(1) - level_set_points) * grad_norm.unsqueeze(1)).sum(-1))
            
            loss_filter_level = ((project_dis_level + project_dis2_level) * weight_level).sum(-1).mean()
            
            
            loss_zero_sdf = torch.abs(sdf_near_points).mean()
            loss_rep = self.lambda_cd * self.ChamferDisL1(near_points.reshape(-1, 3).unsqueeze(0), sample_moved.unsqueeze(0))
            loss = loss_rep + loss_filter_zero + loss_filter_level + loss_zero_sdf
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % 1000 == 0:
                print_log('iter:{:8>d}, loss_filter_zero = {:.6f}, loss_filter_level={:.6f}, rep_loss={:.6f}, loss_zero_sdf={:.6f}, loss = {:.6f}, lr={:.6f} '.
                          format(self.iter_step, loss_filter_zero, loss_filter_level, loss_rep, loss_zero_sdf, loss, self.optimizer.param_groups[0]['lr']),
                          logger=logger)

            if self.iter_step % self.val_freq == 0 and self.iter_step!=0: 
                self.validate_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger)
            if self.iter_step % self.maxiter == 0 and self.iter_step!=0: 
                self.save_checkpoint()


    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))


    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1) 
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        
        self.iter_step = checkpoint['iter_step']
            
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/np_srb.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='gargoyle')
    parser.add_argument('--dataname', type=str, default='gargoyle')
    parser.add_argument('--cls', type=str, default=None)
    parser.add_argument('--lambda_cd', type=float, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.load_checkpoint('ckpt_040000.pth')
        threshs = [-0.001,-0.005] 
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)