import os
# os.environ["OMP_NUM_THREADS"] = "4"
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import jittor as jt
import numpy as np
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from dataset.format import num_joints

from models.skeleton_test_telu import create_model 

from models.metrics import J2J
from jittor.nn import LRScheduler
import math
from jittor.lr_scheduler import CosineAnnealingLR   


jt.flags.use_cuda = 1



def huber_loss(y_true, y_pred, delta=0.2):
    error = y_true - y_pred
    abs_error = jt.abs(error)
    condition = abs_error <= delta
    quadratic_loss = 0.5 * jt.sqr(error)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = jt.where(condition, quadratic_loss, linear_loss)
    return jt.mean(loss)
@jt.single_process_scope()

def validate_one_epoch(epoch, model, val_loader, criterion, best_models,
                       log_message, output_dir):
    """
    被装饰后：
      * 只有 rank==0 真正运行
      * 其它 rank 直接拿到返回值
    注意：函数内部不要再判断 jt.rank
    """
    model.eval()
    val_loss = 0.0
    J2J_loss = 0.0

    with jt.no_grad():
        for data in val_loader:
            vertices, joints = data['vertices'], data['joints']
            joints = joints.reshape(joints.shape[0], -1)
            vertices = vertices.permute(0, 2, 1)
            outputs = model(vertices)
            loss = criterion(outputs, joints)
            val_loss += loss.item()

            for i in range(outputs.shape[0]):
                J2J_loss += J2J(outputs[i].reshape(-1, 3),
                                joints[i].reshape(-1, 3)).item() / outputs.shape[0]

    val_loss /= len(val_loader)
    J2J_loss /= len(val_loader)

    log_message(f"Validation Loss: {val_loss:.4f}  J2J Loss: {J2J_loss:.4f}")

    # 维护 best_models 列表
    if J2J_loss < best_models[-1][0]:
        worst_model_path = best_models[-1][1]
        if worst_model_path and os.path.exists(worst_model_path):
            os.remove(worst_model_path)
            log_message(f"Removed old best model: {os.path.basename(worst_model_path)}")

        new_model_path = os.path.join(output_dir,
                                      f'best_model_epoch_{epoch+1}_loss_{J2J_loss:.4f}.pkl')
        model.save(new_model_path)
        log_message(f"Saved new best model with J2J loss {J2J_loss:.4f} to {new_model_path}")

        best_models[-1] = (J2J_loss, new_model_path)
        best_models.sort(key=lambda x: x[0])

    return best_models
def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    if jt.rank == 0:
        log_message(f"Starting training with parameters: {args}")

    model = create_model(
        model_name=args.model_name,
        output_channels=num_joints*3,
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = nn.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    
    scheduler = CosineAnnealingLR(optimizer, eta_min=args.learning_rate*0.01, T_max=args.epochs)
    criterion = nn.MSELoss()
    L1_loss = nn.L1Loss()
    

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
        sampler=sampler,
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
            sampler=sampler,
            transform=transform,
            random_pose=True,
        )
    else:
        val_loader = None
    

    for epoch in range(start_epoch, args.epochs):
        model.train()
        # num_steps = 0
        train_loss = 0.0
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            vertices, joints = data['vertices'], data['joints']
            vertices = vertices.permute(0, 2, 1)
            outputs = model(vertices)
            joints = joints.reshape(outputs.shape[0], -1)
            loss = huber_loss(outputs, joints)
            
            optimizer.backward(loss)
            optimizer.step()

            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                if jt.rank == 0:
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                            f"Loss: {loss.item():.4f} ")
        
  
        scheduler.step()
        
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        if jt.rank == 0:
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}")  

        if val_loader is not None and ((epoch+5)>= args.epochs):
            if jt.rank==0:
                jt.sync_all(True) 
                new_model_path = os.path.join(args.output_dir, f'best_model_epoch_{epoch+1}.pkl')
                model.save(new_model_path)
          

def main():
    parser = argparse.ArgumentParser(description='Train a point cloud model')
  
    parser.add_argument('--train_data_list', type=str, default='code/data/trainval_list.txt',required=False)
    parser.add_argument('--val_data_list', type=str, default='code/data/trainval_list.txt',required=False)
    parser.add_argument('--data_root', type=str, default='code/data')
    parser.add_argument('--model_name', type=str, default='pct')
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=480)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--random_pose', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output/skeleton')
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
    seed_all(2025)
    main()

