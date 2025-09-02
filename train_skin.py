import os
# os.environ["OMP_NUM_THREADS"] = "4"

import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from dataset.format import num_joints
from models.skin import create_model 
from models.metrics import J2J
from dataset.exporter import Exporter

from jittor.lr_scheduler import CosineAnnealingLR  
import math

jt.flags.use_cuda = 1

def huber_loss(y_true, y_pred, delta=0.3):
    error = y_true - y_pred
    abs_error = jt.abs(error)
    condition = abs_error <= delta
    quadratic_loss = 0.5 * jt.sqr(error)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = jt.where(condition, quadratic_loss, linear_loss)
    return jt.mean(loss)

def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    log_message(f"Starting training with parameters: {args}")
    
    model = create_model(
        model_name=args.model_name,
        output_channels=num_joints*3,
    )
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    scheduler = CosineAnnealingLR(optimizer, eta_min=args.learning_rate*0.01, T_max=args.epochs)
    
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    

    start_epoch = 0
    best_models = [(float('inf'), None)] * 3
    
    
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)

    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
        random_pose=args.random_pose,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
            random_pose=args.random_pose,
        )
    else:
        val_loader = None
    

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_loss_huber_joints = 0.0
        start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']
            outputs_joints, outputs_skin = model(vertices)
            
            loss_mse = criterion_mse(outputs_skin, skin)
            loss_l1 = criterion_l1(outputs_skin, skin)
            loss_huber_joints = huber_loss(outputs_joints, joints)
            loss = loss_mse + loss_l1 + loss_huber_joints
            
            optimizer.backward(loss)
            optimizer.step()
            
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            train_loss_huber_joints += loss_huber_joints.item()
            
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss huber joints: {loss_huber_joints.item():.4f} Loss mse: {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f}")

        scheduler.step()
        
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        train_loss_huber_joints /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss mse: {train_loss_mse:.4f} "
                   f"Train Loss l1: {train_loss_l1:.4f} "
                   f"Train Loss Huber Joints: {train_loss_huber_joints:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")  

        if val_loader is not None and ((epoch + 1) % 30 == 0 or (epoch+10)>= args.epochs): 
            model.eval()
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            J2J_loss = 0.0
            with jt.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                    outputs_joints, outputs_skin = model(vertices)
                    loss_mse = criterion_mse(outputs_skin, skin)
                    loss_l1 = criterion_l1(outputs_skin, skin)
                    
                    val_loss_mse += loss_mse.item()
                    val_loss_l1 += loss_l1.item()
                    for i in range(outputs_joints.shape[0]):
                        J2J_loss += J2J(outputs_joints[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs_joints.shape[0]
            
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: j2j:{J2J_loss:.4f} mse: {val_loss_mse:.4f} l1: {val_loss_l1:.4f}")
            
            if val_loss_l1 < best_models[-1][0]:
                worst_model_path = best_models[-1][1]
                if worst_model_path and os.path.exists(worst_model_path):
                    os.remove(worst_model_path)
                    log_message(f"Removed old best model: {os.path.basename(worst_model_path)}")

                new_model_path = os.path.join(args.output_dir, f'best_model_epoch_{epoch+1}_loss_{val_loss_l1:.4f}.pkl')
                model.save(new_model_path)
                log_message(f"Saved new best model with L1 loss {val_loss_l1:.4f} to {new_model_path}")
                best_models[-1] = (val_loss_l1, new_model_path)
                best_models.sort(key=lambda x: x[0])
            
    log_message(f"Training completed.")

def main():
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    parser.add_argument('--train_data_list', type=str, default='code/data/trainval_list.txt',required=False)
    parser.add_argument('--val_data_list', type=str, default='code/data/trainval_list.txt', required=False)
    parser.add_argument('--data_root', type=str, default='code/data')
    parser.add_argument('--model_name', type=str, default='pct')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--random_pose', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output/skin')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=30)
    parser.add_argument('--val_freq', type=int, default=1)
    
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(3407)
    main()