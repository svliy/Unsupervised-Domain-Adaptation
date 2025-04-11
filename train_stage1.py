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
from losses import WeightedAsymmetricLoss, ContrastiveLossInfoNCE
from description import describe_au

import pdb

logger = logging.getLogger('transfer')

def set_dataset(config):
    logger.info('==> Preparing data...')
    source_dataset = GeneralData(config.source_data_root, config.source_train_list, config, phase="train")
    target_dataset = GeneralData(config.target_data_root, config.target_train_list, config, phase="train")
    val_dataset = GeneralData(config.target_data_root, config.target_test_list, config, phase="test")
    
    logger.info(f"The dataset length:")
    logger.info(f"source-train dataset: {len(source_dataset)}")
    logger.info(f"target-train dataset: {len(target_dataset)}")
    logger.info(f"target-val dataset: {len(val_dataset)}")
    
    source_loader = DataLoader(source_dataset, shuffle=True, drop_last = True,
                               batch_size=config.batch_size,
                               num_workers=config.number_works)
    target_loader = DataLoader(target_dataset, shuffle=True, drop_last = True,
                            batch_size=config.batch_size,
                            num_workers=config.number_works)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last = False,
                            batch_size=config.batch_size,
                            num_workers=config.number_works)

    return source_loader, target_loader, val_loader

def train(epoch, config, model, source_loader, target_loader, criterion, optimizer, scheduler, text_ori):
    # 记录损失
    loss_source_bce = AverageMeter()
    loss_source_cl = AverageMeter()
    loss_all = AverageMeter()
    # 设置模型为train模式
    model.train()
    model.vision_encoder.eval()
    # pdb.set_trace()
    # 计算指标
    statistics_list = None
    # 手动获取数据迭代器
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    source_len = len(source_loader)
    target_len = len(target_loader)
    n = min(len(source_loader), len(target_loader))
    display_interval = n // 5
    
    for batch_idx in tqdm(range(n)):
        
        try:
            source_img, source_label = next(source_iter) # [36, 3, 224, 224], [36, 5]
        except StopIteration:
            source_iter = iter(source_loader)
            source_img, source_label = next(source_iter)

        try:
            target_img, target_label = next(target_iter) # [36, 3, 224, 224], [36, 5]
        except StopIteration:
            target_iter = iter(target_loader)
            target_img, target_label = next(target_iter)
        
        # 将数据移动到GPU上
        if torch.cuda.is_available():
            source_img, source_label = source_img.cuda(), source_label.float().cuda()
            target_img, target_label = target_img.cuda(), target_label.float().cuda()
        
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        source_output = model(source_img, text_ori)
        # 损失计算，临时变量
        source_bce_loss = criterion[0](source_output['logits'].sigmoid(), source_label) # 源域分类损失
        source_cl_loss = criterion[1](source_output['vision_features'], source_output['text_features'], source_label) # 源域对比损失
        
        source_cl_loss = source_cl_loss * config.source_cl_loss_weight
        
        loss = source_bce_loss + source_cl_loss# 整体损失
        # backward pass
        loss.backward()
        # update the parameters
        optimizer.step()
        scheduler.step()
        
        loss_source_bce.update(source_bce_loss.data.item(), config.batch_size)
        loss_source_cl.update(source_cl_loss.data.item(), config.batch_size)
        loss_all.update(loss.data.item(), config.batch_size)
        
        update_list = statistics(source_output['logits'].sigmoid().detach(), source_label.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)
        
        # 损失记录
        if batch_idx == 0 or (batch_idx % display_interval == 0):
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{n} | loss: {loss_all.avg:.6f} | source bce loss: {loss_source_bce.avg:.6f} | source cl loss: {loss_source_cl.avg:.6f}')
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    
    return {
        'loss': loss_all.avg,
        'source_bce_loss': loss_source_bce.avg,
        'source_cl_loss': loss_source_cl.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list
    }
    
def val(epoch, config, model, val_loader, criterion, optimizer, text_ori):
    # 记录损失
    loss_all = AverageMeter()
    model.eval()
    statistics_list = None
    n = len(val_loader)
    display_interval = n // 5
    
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            output = model(images, text_ori)
            # 损失计算，临时变量
            bce_loss = criterion[0](output['logits'].sigmoid(), labels) # 源域分类损失
            loss = bce_loss # 整体损失
            loss_all.update(loss.data.item(), config.batch_size)
            
            update_list = statistics(output['logits'].sigmoid().detach(), labels.detach(), 0.5)
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
    parse.add_argument('--mode', type=str, default='cross', help='source, cross, target')
    args = parse.parse_args()
    
    # 获取配置
    config = set_config(args)
    config = set_outdir(config) # 设置输出路径
    set_seed(config.seed) # 设置随机种子
    set_logger(config) # 设置日志
    dataset_info = set_dataset_info(config)
    
    # 构造数据集
    logger.info(f"********Cross: {config.exp_name}********")
    source_loader, target_loader, val_loader = set_dataset(config)
    
    # 获取weight
    weight = get_weight(config.source_train_list, config.au_list)
    logger.info(f"Weight: {weight}")
    
    if config.train_or_test == 'train':
        # 训练模式
        logger.info("********Train********")
        # 模型
        model = TransferAU(config).cuda()
        # 冻结视觉编码器参数
        logger.info('==> Freeze vision encoder...')
        for name, param in model.named_parameters():
            if 'vision_encoder' in name or 'text_encoder' in name:
                param.requires_grad = False

        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # 损失函数和优化器
        source_bce_loss = WeightedAsymmetricLoss(weight=weight.cuda())
        source_cl_loss = ContrastiveLossInfoNCE()
        criterion = [source_bce_loss, source_cl_loss]
        
        params = list(params) + list(source_cl_loss.parameters())
        
        if config.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=config.lr_init, weight_decay=config.weight_decay)
        # 学习率调度器
        n = min(len(source_loader), len(target_loader))
        iterations_per_epoch = math.ceil(n)
        total_iterations = iterations_per_epoch * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
        
        # Train and val loop    
        best_f1 = 0
        best_epoch = 0
        # get text
        text_ori = describe_au(config.au_list)
        for epoch in range(1, config.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            logger.info("Epoch: [{} | {} LR: {} ]".format(epoch, config.epochs, lr))
            
            logger.info('==> Training...')
            train_output = train(epoch, config, model, source_loader, target_loader, criterion, optimizer, scheduler, text_ori)
            logger.info({'Epoch: {}  train_f1: {:.2f} train_loss: {:.6f}  bce_loss: {:.6f} cl_loss: {:.6f}'.format(epoch, 100.*train_output['mean_f1_score'], train_output['loss'], train_output['source_bce_loss'], train_output['source_cl_loss'])})
            logger.info({'Train F1-score-list:'})
            logger.info(dataset_info(train_output['f1_score_list']))
            
            logger.info('==> Validation...')
            val_output = val(epoch, config, model, val_loader, criterion, optimizer, text_ori)
            logger.info({'Epoch: {}  test_f1: {:.2f} test_loss: {:.6f}'.format(epoch, 100.*val_output['mean_f1_score'], val_output['loss'])})
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
                    'source_cl_state_dict': source_cl_loss.state_dict(),
                }
                torch.save(checkpoint, os.path.join(config['outdir'], 'best_model_fold' + str(config.fold) + '.pth'))
            torch.save(checkpoint, os.path.join(config['outdir'], 'epoch_' + str(epoch) + '_model_fold' + str(config.fold) + '.pth'))
            logger.info(f"Best f1: {best_f1} at epoch {best_epoch}")
    
    elif config.train_or_test == 'test':
        # 测试模式
        logger.info("********Test********")
        # 模型
        # pdb.set_trace()
        model = TransferAU(config).cuda()
        # 加载模型
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        pdb.set_trace()
                
        # 冻结视觉编码器参数
        logger.info('==> Freeze vision encoder...')
        for name, param in model.named_parameters():
            param.requires_grad = False

        params = filter(lambda p: p.requires_grad, model.parameters())
        
        # 损失函数和优化器
        source_cl_loss = ContrastiveLossInfoNCE()
        criterion = [WeightedAsymmetricLoss(weight=weight.cuda()), source_cl_loss]
        
        params = list(params) + list(source_cl_loss.parameters())
        
        if config.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=config.lr_init, weight_decay=config.weight_decay)
        # 学习率调度器
        n = min(len(source_loader), len(target_loader))
        iterations_per_epoch = math.ceil(n)
        total_iterations = iterations_per_epoch * config.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
        
        # Train and val loop    
        best_f1 = 0
        best_epoch = 0
        # get text
        text_ori = describe_au(config.au_list)
            
        logger.info('==> Validation...')
        val_output = val(9999, config, model, val_loader, criterion, optimizer, text_ori)
        logger.info({'Epoch: {}  test_f1: {:.2f} test_loss: {:.6f}'.format(9999, 100.*val_output['mean_f1_score'], val_output['loss'])})
        logger.info({'Val F1-score-list:'})
        logger.info(dataset_info(val_output['f1_score_list']))