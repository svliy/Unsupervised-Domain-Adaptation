import logging
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.transfer_au import TransferNet
from dataset import GeneralData
from utils import *
from loss import ContrastiveLossInfoNCE, ContrastiveLossForSource
from description import describe_au

import pdb
logger = logging.getLogger('transfer')

def set_dataset(config):
    
    logger.info('****************[Dataset and Dataloader]****************')
    
    source_train_dataset = GeneralData(config.source_data_root, config.source_train_list, config, phase="train")
    target_dataset = GeneralData(config.target_data_root, config.target_train_list, config, phase="train")
    val_dataset = GeneralData(config.target_data_root, config.target_test_list, config, phase="val")
    
    source_name, target_name = config.exp_name.split('2')
    logger.info(f"The dataset length:")
    logger.info(f"****************[source:{source_name}]****************")
    logger.info(f"source-train dataset: {len(source_train_dataset)}")
    logger.info(f"****************[target:{target_name}]****************")
    logger.info(f"target-train dataset: {len(target_dataset)}")
    logger.info(f"target-val dataset: {len(val_dataset)}")
    
    source_train_loader = DataLoader(source_train_dataset, shuffle=True, drop_last = False,
                                     batch_size=config.batch_size,
                                     num_workers=config.number_works)
    target_loader = DataLoader(target_dataset, shuffle=True, drop_last = False,
                               batch_size=config.batch_size,
                               num_workers=config.number_works)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last = False,
                            batch_size=config.batch_size,
                            num_workers=config.number_works)
    dataloaders = {
        'source_train': source_train_loader,
        'target_train': target_loader,
        'target_val': val_loader,
    }
    
    return dataloaders


def set_optimizer(config, params):
    
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(params, lr=config.lr_init, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay)
        
    return optimizer


def train(epoch, config, model, dataloaders, criterion, optimizer, scheduler=None):
    # 记录损失
    loss_all = AverageMeter()
    loss_target_bce = AverageMeter()
    loss_target_cl = AverageMeter()

    model.train()
    model.freeze_bn()
    
    statistics_list = None
    target_loader = dataloaders['target_train']
    len_dataloader = len(target_loader)
    
    for batch_idx, (target_img, target_label, target_img_name) in enumerate(tqdm(target_loader)):
        if torch.cuda.is_available():
            target_img, target_label = target_img.cuda(), target_label.float().cuda()
        
        optimizer.zero_grad()
        target_output = model(target_img)
        target_bce_loss = criterion['target_bce_loss'](target_output['logits'], target_label) # 源域分类损失
        target_cl_loss = criterion['target_cl_loss'](target_output['au_vision_features'], target_output['au_text_features'], target_label) # 源域对比损失
        loss = target_bce_loss + config.target_cl_loss_weight*target_cl_loss # 整体损失
        loss.backward()
        optimizer.step()
        # print(f'可学习上下文向量：{model.prompt_learner.ctx[:,:5]}')
        
        loss_target_bce.update(target_bce_loss.data.item(), config.batch_size)
        loss_target_cl.update(target_cl_loss.data.item(), config.batch_size)
        loss_all.update(loss.data.item(), config.batch_size)
        
        update_list = statistics(target_output['logits'].sigmoid().detach(), target_label.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)
        
        if batch_idx == 0 or (batch_idx % (len_dataloader // 10) == 0):
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len_dataloader} | loss: {loss_all.avg:.6f} | target_bce_loss: {loss_target_bce.avg} | target_cl_loss: {loss_target_cl.avg}')
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return {
        'loss': loss_all.avg,
        'target_bce_loss': loss_target_bce.avg,
        'target_cl_loss': loss_target_cl.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list,
        'mean_acc': mean_acc,
        'acc_list': acc_list
    }


def val(epoch, config, model, dataloaders, criterion=None):
    # 记录损失
    loss_all = AverageMeter()
    loss_target_bce = AverageMeter()
    loss_target_cl = AverageMeter()
    model.eval()
    statistics_list = None
    val_loader = dataloaders['target_val']
    len_dataloader = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels, img_name) in enumerate(tqdm(val_loader)):        
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            target_output = model(images) # forward pass
            target_bce_loss = criterion['target_bce_loss'](target_output['logits'], labels) # 源域分类损失
            target_cl_loss = criterion['target_cl_loss'](target_output['au_vision_features'], target_output['au_text_features'], labels) # 源域对比损失
            loss = target_bce_loss + config.target_cl_loss_weight*target_cl_loss # 整体损失
            
            loss_target_bce.update(target_bce_loss.data.item(), config.batch_size)
            loss_target_cl.update(target_cl_loss.data.item(), config.batch_size)
            loss_all.update(loss.data.item(), config.batch_size)
            
            update_list = statistics(target_output['logits'].sigmoid().detach(), labels.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
            
            if batch_idx == 0 or (batch_idx % (len_dataloader // 4) == 0):
                logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len_dataloader} | loss: {loss_all.avg:.6f} | target_bce_loss: {loss_target_bce.avg} | target_cl_loss: {loss_target_cl.avg}')
    
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return {
        'loss': loss_all.avg,
        'target_bce_loss': loss_target_bce.avg,
        'target_cl_loss': loss_target_cl.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list,
        'mean_acc': mean_acc,
        'acc_list': acc_list
    }

def main(args):

    config = set_config(args) # 获取配置
    config = set_outdir(config) # 设置输出路径
    set_seed(config.seed) # 设置随机种子
    set_logger(config) # 设置日志
    
    dataset_info = set_dataset_info(config) # 设置数据集配置信息
    # 构造数据集
    logger.info(f"****************[{config.mode}]:[{config.exp_name}]****************")
    dataloaders = set_dataset(config)
    
    # 获取weight
    source_weight, source_pos_weight = get_weight(config.source_train_list, config.au_list)
    target_weight, target_pos_weight  = get_weight(config.target_train_list, config.au_list)
    logger.info(f"Source Weight: {source_weight}")
    logger.info(f"Source Pos Weight: {source_pos_weight}")
    logger.info(f"Target Weight: {target_weight}")
    logger.info(f"Target Pos Weight: {target_pos_weight}")
    
    if config.train_or_test == 'train':
        logger.info("****************[Train Mode]****************")
        
        model = TransferNet(config)
        model = model.cuda()
        # 模型参数量计算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_millions = total_params / 1_000_000.0
        trainable_params_millions = trainable_params / 1_000_000.0
        logger.info(f"模型总参数量: {total_params} ({total_params_millions:.2f} M)") # 保留两位小数
        logger.info(f"可训练参数量: {trainable_params} ({trainable_params_millions:.2f} M)") # 保留两位小数
        # criterion
        criterion = {
            'target_bce_loss': nn.BCEWithLogitsLoss(weight=target_weight.float().cuda(), pos_weight=target_pos_weight.float().cuda()),
            'target_cl_loss': ContrastiveLossForSource(initial_temperature=0.07),
        }
        # optimizer
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        loss_params = criterion['target_cl_loss'].parameters()
        all_params = list(model_params) + list(loss_params) # 合并所有需要优化的参数
        optimizer = set_optimizer(config, all_params)
        # scheduler
        # iterations_per_epoch = len(dataloaders['source_train'])
        # total_iterations = iterations_per_epoch * config.epochs
        # logger.info(f"The total iterations is {total_iterations}: epoch({config.epochs}) * iterations_per_epoch({iterations_per_epoch})")
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations)
        scheduler = None
        
        # Train and val loop    
        best_f1 = 0
        best_epoch = 0
        config.stop = 0 # early stop
        
        for epoch in range(1, config.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            logger.info("Epoch: [{:2d}/{:2d}], lr: {}".format(epoch, config.epochs, lr))
            
            logger.info('==> Training...')
            train_output = train(epoch, config, model, dataloaders, criterion, optimizer, scheduler)
            logger.info({'Epoch: [{:2d}/{:2d}], train_f1: {:.2f}, train_acc: {:.2f}, train_loss: {:.6f}, bce_loss: {:.6f}, cl_loss: {:.6f}'.format(epoch, config.epochs, 100.*train_output['mean_f1_score'], 100.*train_output['mean_acc'], train_output['loss'], train_output['target_bce_loss'], train_output['target_cl_loss'])})
            logger.info({'Train F1-score-list:'})
            logger.info(dataset_info(train_output['f1_score_list']))
            logger.info({'Train Acc-list:'})
            logger.info(dataset_info(train_output['acc_list']))

            logger.info('==> Validation...')
            val_output = val(epoch, config, model, dataloaders, criterion)
            logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}, test_loss: {:.6f}, bce_loss: {:.6f}, cl_loss: {:.6f}'.format(epoch, config.epochs, 100.*val_output['mean_f1_score'], 100.*val_output['mean_acc'], val_output['loss'], val_output['target_bce_loss'], val_output['target_cl_loss'])})
            logger.info({'Val F1-score-list:'})
            logger.info(dataset_info(val_output['f1_score_list']))
            logger.info({'Val Acc-list:'})
            logger.info(dataset_info(val_output['acc_list']))

            current_f1 = val_output['mean_f1_score']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                config.stop = 0
                checkpoint = {
                    'state_dict': model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(config['outdir'], 'E' + str(epoch) + '_model_fold' + str(config.fold) + '.pth'))
            logger.info(f"Best f1: {best_f1} at epoch {best_epoch}")
            config.stop = config.stop + 1
            if config.stop > config.early_stop:
                break
            
    elif config.train_or_test == 'test':
        logger.info("****************[Test Mode]****************")
        
        model = TransferNet(config)
        model = model.cuda()
        # 加载模型
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        # criterion
        criterion = {
            'source_bce_loss': nn.BCEWithLogitsLoss(weight=source_weight.float().cuda(), pos_weight=source_pos_weight.float().cuda()),
            'source_cl_loss': ContrastiveLossInfoNCE(),
            'target_bce_loss': nn.BCEWithLogitsLoss(),
        }
        
        text_ori = describe_au(config.au_list)
        logger.info('==> Validation For Target Val...')
        val_output = val(9999, config, model, dataloaders, criterion, text_ori)
        logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}, test_loss: {:.6f}, bce_loss: {:.6f}, cl_loss: {:.6f}'.format(9999, config.epochs, 100.*val_output['mean_f1_score'], 100.*val_output['mean_acc'], val_output['loss'], val_output['bce_loss'], val_output['cl_loss'])})
        logger.info({'Val F1-score-list:'})
        logger.info(dataset_info(val_output['f1_score_list']))
        logger.info({'Val Acc-list:'})
        logger.info(dataset_info(val_output['acc_list']))
        
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--config_file', type=str, default='bp4d2gft',
                       help='Path for the configuration file.')
    parse.add_argument('--mode', type=str, default='target', choices=['source', 'cross', 'target'],
                       help='Mode of operation: choose from "source", "cross", or "target".')
    args = parse.parse_args()
    
    main(args)
    
    