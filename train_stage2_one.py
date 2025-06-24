import logging
import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.transfer_one import TransferNet
from dataset import GeneralData, PseudoLabelDataset
from utils import *
from loss import ContrastiveLossForSource, MMD_loss
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


def train(epoch, config, model, dataloaders, criterion, optimizer, scheduler, ema_model):
    loss_all = AverageMeter()
    loss_source_bce = AverageMeter()
    loss_target_cl = AverageMeter()
    loss_transfer = AverageMeter()
    
    model.train()
    model.freeze_bn()
    
    statistics_list = None
    source_loader = dataloaders['source_train']
    # target_loader = dataloaders['pseudo_label_loader']
    target_loader = dataloaders['target_train']
    
    source_train_iter = iter(source_loader)
    
    len_dataloader = len(target_loader)
    
    for batch_idx, (target_img, target_truth_label, target_img_name) in enumerate(tqdm(target_loader)):
        # --- 获取源域数据 ---
        try:
            source_img, source_label, source_img_name = next(source_train_iter)
        except StopIteration:
            # 如果源域迭代器耗尽，重新创建它
            source_train_iter = iter(source_loader)
            source_img, source_label, source_img_name = next(source_train_iter)
        if torch.cuda.is_available():
            # target_img, target_truth_label, target_pseudo_label = target_img.cuda(), target_truth_label.float().cuda(), target_pseudo_label.float().cuda()
            target_img, target_truth_label = target_img.cuda(), target_truth_label.float().cuda()
            source_img, source_label = source_img.cuda(), source_label.float().cuda()
        
        optimizer.zero_grad()
        source_output = model(source_img)
        target_output = model(target_img)
        
        source_bce_loss = criterion['source_bce_loss'](source_output['logits'], source_label) # 源域分类损失
        # target_cl_loss = criterion['target_cl_loss'](target_output['au_vision_features'], target_output['au_text_features']) # 目标域对比学习损失        
        # source_bce_loss = torch.tensor(0.0).cuda()
        target_cl_loss = torch.tensor(0.0).cuda()
        
        transfer_loss = criterion['transfer_loss'](torch.flatten(source_output['au_vision_features'], 1),
                                                   torch.flatten(target_output['au_vision_features'], 1))
        
        loss = source_bce_loss + config.transfer_loss_weight*transfer_loss + config.target_cl_loss_weight*target_cl_loss # 整体损失
        
        loss.backward()
        optimizer.step()
        ema_model.update(model) # 更新模型参数
        
        loss_source_bce.update(source_bce_loss.data.item(), config.batch_size)
        loss_target_cl.update(target_cl_loss.data.item(), config.batch_size)
        loss_transfer.update(transfer_loss.data.item(), config.batch_size)
        loss_all.update(loss.data.item(), config.batch_size)
        
        update_list = statistics(target_output['logits'].sigmoid().detach(), target_truth_label.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)
        
        if batch_idx == 0 or (batch_idx % (len_dataloader // 10) == 0):
            logger.info(f'Epoch: {epoch} | Batch: {batch_idx}/{len_dataloader} | loss: {loss_all.avg:.6f} | source_bce_loss: {loss_source_bce.avg} | target_cl_loss: {loss_target_cl.avg} | transfer_loss: {loss_transfer.avg}')
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return {
        'loss': loss_all.avg,
        'source_bce_loss': loss_source_bce.avg,
        'target_cl_loss': loss_target_cl.avg,
        'transfer_loss': loss_transfer.avg,
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list,
        'mean_acc': mean_acc,
        'acc_list': acc_list
    }

   
def val(epoch, config, model, dataloaders, criterion):
    # 记录损失
    model.eval()
    statistics_list = None
    val_loader = dataloaders['target_val']
    
    with torch.no_grad():
        for batch_idx, (images, labels, imag_name) in enumerate(tqdm(val_loader)):        
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            output = model(images) # forward pass
            update_list = statistics(output['logits'].sigmoid().detach(), labels.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return {
        'mean_f1_score': mean_f1_score,
        'f1_score_list': f1_score_list,
        'mean_acc': mean_acc,
        'acc_list': acc_list
    }


def pseudo_label(epoch, config, dataloaders, model, ema_model):
    model.eval()
    enable_dropout(model)
    ema_model.eval()
    enable_dropout(ema_model)
    # model_outputs = []
   
    target_train_loader = dataloaders['target_train']
    num_samples = len(target_train_loader.dataset)
    
    truth_labels = np.zeros((num_samples, config.class_num), dtype=np.float32)
    out_std_list = np.zeros((num_samples, config.class_num), dtype=np.float32)
    out_prob_list = np.zeros((num_samples, config.class_num), dtype=np.float32)
    images_list = np.empty(num_samples, dtype=object)
    
    current_idx = 0
    with torch.no_grad():
        for batch_idx, (images, labels, img_name) in enumerate(tqdm(target_train_loader)):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.float().cuda()
            
            current_batch_size = images.shape[0]
            
            out_prob = []
            for _ in range(config.f_pass):
                output = ema_model(images)
                logits = output['logits'].detach()
                out_prob.append(torch.sigmoid(logits)) # for selecting positive pseudo-labels
            
            out_prob = torch.stack(out_prob) # [f_pass, 256, 10]
            out_std = torch.std(out_prob, dim=0) # [256, 10]
            out_prob = torch.mean(out_prob, dim=0)
            
            start_idx = current_idx
            end_idx = start_idx + current_batch_size
            
            # pseudo_label_tmp = (out_std<=config.kappa_p)*(out_prob>=config.tau_p)
            # pseudo_label_tmp = pseudo_label_tmp.float()
            
            # logits = output['logits'].detach()
            # model_outputs.append(torch.sigmoid(logits).cpu().numpy())
            out_std_list[start_idx:end_idx] = out_std.detach().cpu().numpy()
            out_prob_list[start_idx:end_idx] = out_prob.detach().cpu().numpy()
            images_list[start_idx:end_idx] = list(img_name)
            truth_labels[start_idx:end_idx] = labels.detach().cpu().numpy()
            
            current_idx = end_idx
            
            del images, labels, output, logits, out_std, out_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # if batch_idx == 1:
                # break
            # pseudo_labels.append(pseudo_label_tmp.detach().cpu().numpy())

        # model_outputs = np.concatenate(model_outputs) # (108538, 10)
        # pseudo_labels = (model_outputs>=config.tau_p).astype(np.float32) # (108538, 10)
        # out_std_list = np.concatenate(out_std_list)
        # out_prob_list = np.concatenate(out_prob_list)
        # pseudo_labels = np.concatenate(pseudo_labels)
        # images_list = np.concatenate(images_list) # (108538, 10)
        # truth_labels = np.concatenate(truth_labels) # (108538, 10)
        
        
        
        # 完全正确的标签
        
        # X, y_t, y_p, img_name = next(iter(pseudo_label_loader))
        # y_p = y_p.detach().cpu().numpy()
        # y_t = y_t.detach().cpu().numpy()
        # completely_correct_samples_mask = np.all(y_p == y_t, axis=1)
        # y_p[completely_correct_samples_mask]
        
        # 获取完全正确的伪标签
        # pseudo_labels[completely_correct_samples_mask]
        def get_acc_pseudo_label(p_uncertainty, p_confidence):
            pseudo_labels = (out_std_list<=p_uncertainty)*(out_prob_list>=p_confidence)
            completely_correct_samples_mask = np.all(pseudo_labels == truth_labels, axis=1)
            num_completely_correct_samples = np.sum(completely_correct_samples_mask)
            logger.info(f"Epoch[{epoch}]: Number of completelpy correct samples: {num_completely_correct_samples}")
            logger.info(f"Epoch[{epoch}]: Proportion of completely correct samples: {num_completely_correct_samples/len(pseudo_labels)}")
            # 每个AU中预测的准确度
            correct_predictions = (pseudo_labels == truth_labels).astype(np.float32) # (108538, 10)
            accuracy_per_au = np.mean(correct_predictions, axis=0) # (10,)
            logger.info(f"Epoch[{epoch}]: accuracy_per_au: {accuracy_per_au}")
            return pseudo_labels, num_completely_correct_samples/len(pseudo_labels)
        
        
        kappa_p_list = [0.005, 0.01, 0.02, 0.05, 0.10]
        tau_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        acc_matrix = np.zeros((len(kappa_p_list), len(tau_p_list)))
        
        for i, k_p in enumerate(kappa_p_list):
            for j, t_p in enumerate(tau_p_list):
                logger.info(f"kappa_p: {k_p} | tau_p: {t_p}")
                _, acc_item = get_acc_pseudo_label(k_p, t_p)
                acc_matrix[i][j] = acc_item
                
        pseudo_labels, _ = get_acc_pseudo_label(config.kappa_p, config.tau_p)
        pseudo_label_dataset = PseudoLabelDataset(images_list, truth_labels, pseudo_labels, config)
        pseudo_label_loader = DataLoader(pseudo_label_dataset,
                                         shuffle=False,
                                         drop_last = False,
                                         batch_size=config.batch_size,
                                         num_workers=config.number_works)
        file_path_npy = f"{config['outdir']}/acc_matrix.npy"
        np.save(file_path_npy, acc_matrix)
        # pdb.set_trace()
        # if True:
        #     # 每个类别正确预测为正例的数量 (真阳性 TP)。
        #     n_correct_pos = (truth_labels*pseudo_labels).sum(0)
            
        #     # 每个类别预测为正例的总数 (TP + 假阳性 FP)。
        #     n_pred_pos = ((pseudo_labels==1)).sum(0)
        #     # 每个类别真实为正例的总数 (TP + 假阴性 FN)。
        #     n_true_pos = truth_labels.sum(0)
        #     OP = n_correct_pos.sum()/n_pred_pos.sum() # 0.5418174305699023
        #     CP = np.nanmean(n_correct_pos/n_pred_pos) # 0.4150217363184052
        #     OR = n_correct_pos.sum()/n_true_pos.sum() # 0.54179823
        #     CR = np.nanmean(n_correct_pos/n_true_pos) # 0.4150001

        #     # auc = np.zeros(labels.shape[1])
        #     # for i in range(labels.shape[1]):
        #         # auc[i] = metrics.roc_auc_score(labels[:,i], pseudo_labels[:,i])
        #     # AUC = np.nanmean(auc)
        #     logger.info('Train: ')
        #     # logger.info(' AUC: %.3f'%AUC)
        #     logger.info(' OP: %.3f'%OP)
        #     logger.info(' CP: %.3f'%CP)
        #     logger.info(' OR: %.3f'%OR)
        #     logger.info(' CR: %.3f'%CR)
        
        return pseudo_label_loader


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
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        ema_m = ModelEma(model, config.ema_decay)
        
        # 模型参数量计算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_millions = total_params / 1_000_000.0
        trainable_params_millions = trainable_params / 1_000_000.0
        print(f"模型总参数量: {total_params} ({total_params_millions:.2f} M)") # 保留两位小数
        print(f"可训练参数量: {trainable_params} ({trainable_params_millions:.2f} M)") # 保留两位小数
        
        # criterion
        criterion = {
            'source_bce_loss': nn.BCEWithLogitsLoss(weight=source_weight.float().cuda(), pos_weight=source_pos_weight.float().cuda()),
            'target_bce_loss': nn.BCEWithLogitsLoss(),
            'target_cl_loss': ContrastiveLossForSource(initial_temperature=0.07),
            'transfer_loss': MMD_loss(),
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
        
        # logger.info('==> Validation...')
        # val_output = val(9999, config, model, dataloaders, criterion)
        # logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}'.format(9999, config.epochs, 100.*val_output['mean_f1_score'], 100.*val_output['mean_acc'])})
        # logger.info({'Val F1-score-list:'})
        # logger.info(dataset_info(val_output['f1_score_list']))
        # logger.info({'Val Acc-list:'})
        # logger.info(dataset_info(val_output['acc_list']))
        
        logger.info('==> EMA Validation...')
        ema_val_output = val(9999, config, ema_m.module, dataloaders, criterion)
        logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}'.format(9999, config.epochs, 100.*ema_val_output['mean_f1_score'], 100.*ema_val_output['mean_acc'])})
        logger.info({'Val F1-score-list:'})
        logger.info(dataset_info(ema_val_output['f1_score_list']))
        logger.info({'Val Acc-list:'})
        logger.info(dataset_info(ema_val_output['acc_list']))
            
        # pdb.set_trace()
        for epoch in range(1, config.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            logger.info("Epoch: [{:2d}/{:2d}], lr: {}".format(epoch, config.epochs, lr))
            
            # logger.info('==> Pseudo labeling...')
            # pseudo_label_loader = pseudo_label(epoch, config, dataloaders, model, ema_m.module)
            # dataloaders['pseudo_label_loader'] = pseudo_label_loader
            
            logger.info('==> Training...')
            train_output = train(epoch, config, model, dataloaders, criterion, optimizer, scheduler, ema_m)
            logger.info({'Epoch: [{:2d}/{:2d}], train_f1: {:.2f}, train_acc: {:.2f}, train_loss: {:.6f}, source_bce_loss: {:.6f}, target_cl_loss: {:.6f}, transfer_loss: {:.6f}'.format(epoch, config.epochs, 100.*train_output['mean_f1_score'], 100.*train_output['mean_acc'], train_output['loss'], train_output['source_bce_loss'], train_output['target_cl_loss'], train_output['transfer_loss'])})
            logger.info({'Train F1-score-list:'})
            logger.info(dataset_info(train_output['f1_score_list']))
            logger.info({'Train Acc-list:'})
            logger.info(dataset_info(train_output['acc_list']))

            logger.info('==> Validation...')
            val_output = val(epoch, config, model, dataloaders, criterion)
            logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}'.format(epoch, config.epochs, 100.*val_output['mean_f1_score'], 100.*val_output['mean_acc'])})
            logger.info({'Val F1-score-list:'})
            logger.info(dataset_info(val_output['f1_score_list']))
            logger.info({'Val Acc-list:'})
            logger.info(dataset_info(val_output['acc_list']))
            
            logger.info('==> EMA Validation...')
            ema_val_output = val(epoch, config, ema_m.module, dataloaders, criterion)
            logger.info({'Epoch: [{:2d}/{:2d}], test_f1: {:.2f}, test_acc: {:.2f}'.format(epoch, config.epochs, 100.*ema_val_output['mean_f1_score'], 100.*ema_val_output['mean_acc'])})
            logger.info({'Val F1-score-list:'})
            logger.info(dataset_info(ema_val_output['f1_score_list']))
            logger.info({'Val Acc-list:'})
            logger.info(dataset_info(ema_val_output['acc_list']))

            current_f1 = val_output['mean_f1_score']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                checkpoint = {
                    'state_dict': model.state_dict(),
                }
                config.stop = 0
                torch.save(checkpoint, os.path.join(config['outdir'], 'best_model_fold' + str(config.fold) + '.pth'))
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
            'target_bce_loss': nn.BCEWithLogitsLoss(),
            'target_cl_loss': ContrastiveLossForSource(initial_temperature=0.07),
        }
        
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
    parse.add_argument('--mode', type=str, default='cross', choices=['source', 'cross', 'target'],
                       help='Mode of operation: choose from "source", "cross", or "target".')
    args = parse.parse_args()
    main(args)
    
    