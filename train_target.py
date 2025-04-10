import logging
import argparse
import math

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.uda import TransferAU
from dataset import GeneralData
from utils import *
from losses import WeightedAsymmetricLoss

import pdb

logger = logging.getLogger('transfer')

def set_dataset(config):
    logger.info('==> Preparing data...')
    target_dataset = GeneralData(config.target_data_root, config.target_train_list, config, phase="train")
    val_dataset = GeneralData(config.target_data_root, config.target_test_list, config, phase="test")
    
    logger.info(f"The dataset length:")
    logger.info(f"target-train dataset: {len(target_dataset)}")
    logger.info(f"target-val dataset: {len(val_dataset)}")
    
    target_loader = DataLoader(target_dataset, shuffle=True, drop_last = True,
                            batch_size=config.batch_size,
                            num_workers=config.number_works)
    val_loader = DataLoader(val_dataset, shuffle=True, drop_last = True,
                            batch_size=config.batch_size,
                            num_workers=config.number_works)

    return target_loader, val_loader

def train(epoch, config, model, target_loader, criterion, optimizer, scheduler):
    # 记录损失
    loss_bce = AverageMeter()
    loss_all = AverageMeter()
    # 设置模型为train模式
    model.train()
    model.vision_encoder.eval()
    # pdb.set_trace()
    # 计算指标
    statistics_list = None
    # 手动获取数据迭代器
    n = len(target_loader)
    display_interval = n // 5
    
    for batch_idx, (images, labels) in enumerate(tqdm(target_loader)):
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.float().cuda()

        # zero gradients
        optimizer.zero_grad()
        # forward pass
        output = model(images)
        # 损失计算，临时变量
        bce_loss = criterion[0](output.sigmoid(), labels) # 源域分类损失
        loss = bce_loss # 整体损失
        # backward pass
        loss.backward()
        # update the parameters
        optimizer.step()
        scheduler.step()
        
        loss_bce.update(bce_loss.data.item(), config.batch_size)
        loss_all.update(loss.data.item(), config.batch_size)
        
        update_list = statistics(output.sigmoid().detach(), labels.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)
        
        # 损失记录
        if batch_idx == 0 or (batch_idx % display_interval == 0):
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{n} | loss: {loss_all.avg:.6f} | bce loss: {loss_bce.avg:.6f}')
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    
    return {
        'loss': loss_all.avg,
        'bce_loss': loss_bce.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list
    }
    
def val(epoch, config, model, val_loader, criterion, optimizer):
    # 记录损失
    loss_all = AverageMeter()
    model.eval()
    statistics_list = None
    n = len(val_loader)
    display_interval = n // 5
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # 损失计算，临时变量
            bce_loss = criterion[0](output.sigmoid(), labels) # 源域分类损失
            loss = bce_loss # 整体损失
            loss_all.update(loss.data.item(), config.batch_size)
            
            update_list = statistics(output.sigmoid().detach(), labels.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
            # 损失记录
            # if batch_idx == 0 or (batch_idx % display_interval == 0):
            #     logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{n} | loss: {loss_all.avg:.6f}')
    
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    
    return {
        'loss': loss_all.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list
    }

if __name__ == '__main__':
    # 命令行设置
    parse = argparse.ArgumentParser()
    parse.add_argument('--config_name', type=str, default='bp4d2disfa', help='name for config file')
    parse.add_argument('--mode', type=str, default='target', help='source, cross, target')
    args = parse.parse_args()
    
    # 获取配置
    config = set_config(args)
    config = set_outdir(config) # 设置输出路径
    set_seed(config.seed) # 设置随机种子
    set_logger(config) # 设置日志
    dataset_info = set_dataset_info(config)
    
    # 构造数据集
    logger.info("********Target********")
    target_loader, val_loader = set_dataset(config)
    # 获取weight
    weight = get_weight(config.target_train_list, config.au_list)
    logger.info(f"Weight: {weight}")
    
    # 模型
    model = TransferAU(config).cuda()
    # 冻结视觉编码器参数
    logger.info('==> Freeze vision encoder...')
    for name, param in model.named_parameters():
        if 'vision_encoder' in name:
            param.requires_grad = False

    params = filter(lambda p: p.requires_grad, model.parameters())
    # 损失函数和优化器
    criterion = [WeightedAsymmetricLoss(weight=weight.cuda())]
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=config.lr_init, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=config.lr_init, momentum=0.9, weight_decay=5e-4)
    
    # 学习率调度器
    iterations_per_epoch = math.ceil(len(target_loader))
    total_iterations = iterations_per_epoch * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
    
    # Train and val loop    
    best_f1 = 0
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        logger.info("Epoch: [{} | {} LR: {} ]".format(epoch, config.epochs, lr))
        
        logger.info('==> Training...')
        train_output = train(epoch, config, model, target_loader, criterion, optimizer, scheduler)
        logger.info({'Epoch: {}  train_loss: {:.6f}  wa_loss: {:.6f} train_f1: {:.2f}'.format(epoch, train_output['loss'], train_output['bce_loss'], 100.*train_output['mean_f1_score'])})
        logger.info({'Train F1-score-list:'})
        logger.info(dataset_info(train_output['f1_score_list']))
        
        logger.info('==> Validation...')
        val_output = val(epoch, config, model, val_loader, criterion, optimizer)
        logger.info({'Epoch: {}  test_loss: {:.6f} test_f1: {:.2f}'.format(epoch, val_output['loss'], 100.*val_output['mean_f1_score'])})
        logger.info({'Val F1-score-list:'})
        logger.info(dataset_info(val_output['f1_score_list']))

        current_f1 = val_output['mean_f1_score']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config['outdir'], 'best_model_fold' + str(config.fold) + '.pth'))
        
        logger.info(f"Best f1: {best_f1} at epoch {best_epoch}")