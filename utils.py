import os
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import yaml
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from math import cos, pi
from datetime import datetime
from easydict import EasyDict as edict

import pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def statistics(pred, y, thresh):
    # pdb.set_trace()
    """
        pred: [64, 8]
        y: [64, 8]
        
        return:
            statistics_list
            [
                {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
                {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
            ]
    """
    batch_size = pred.size(0) # 64
    # 标签的数量：8
    class_nb = pred.size(1) 
    pred = pred >= thresh
    pred = pred.long()
    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(batch_size):
            # pdb.set_trace()
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list
    # BP4D: 12    
    # DISFA: 8
    assert len(old_list) == len(new_list)

    # 8
    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc(statistics_list):
    acc_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        acc = (TP+TN)/(TP+TN+FP+FN)
        acc_list.append(acc)
    mean_acc_score = sum(acc_list) / len(acc_list)

    return mean_acc_score, acc_list


def BP4D_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}
    return infostr

def GFT_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f}  AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9])}
    return infostr

def DISFA_infolist(list):
    infostr = {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f}  AU6: {:.2f} AU9: {:.2f} AU12: {:.2f}  AU25: {:.2f} AU26: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7])}
    return infostr

def adjust_learning_rate(optimizer, epoch, epochs, init_lr, iteration, num_iter):

    current_iter = iteration + epoch * num_iter
    max_iter = epochs * num_iter
    lr = init_lr * (1 + cos(pi * current_iter / max_iter)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class image_train(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img

class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img

def load_state_dict(model,path):
    checkpoints = torch.load(path, map_location = torch.device('cpu'))
    state_dict = checkpoints['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict, strict=False)
    return model

def set_config(args):
    with open(f'config/{args.config_name}.yaml', 'r') as f:
        config = edict(yaml.safe_load(f))
        config.config_name = args.config_name
        config.mode = args.mode
    return config

def set_seed(seed: int):
    # set seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logging.info('{} is created'.format(dir_name))

def set_outdir(config):
    default_outdir = 'results'
    if 'timedir' in config:
        timestr = datetime.now().strftime('%d-%m-%Y_%I_%M-%S_%p')
        outdir = os.path.join(default_outdir, config.exp_name,timestr)
    else:
        outdir = os.path.join(default_outdir, config.exp_name)
        prefix = str(config.stage)+'_'+str(config.train_or_test)+'_'+str(config.mode)+'_fold_'+str(config.fold)+'_seed_'+str(config.seed)+'_date_'+datetime.now().strftime('%Y-%m-%d_%I_%M-%S_%p')
        outdir = os.path.join(outdir,prefix)
    ensure_dir(outdir)
    config['outdir'] = outdir
    shutil.copyfile(f"./config/{config.config_name}.yaml", os.path.join(outdir, f'{config.config_name}.yaml'))

    return config

def set_logger(config):
    # 定义一个name为transfer的logger
    logger = logging.getLogger('transfer')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Logging to file
    log_path = os.path.join(os.getcwd(), config['outdir'], 'train.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s >>> %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s >>> %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(stream_handler)
    
    logger.info('writting logs to file {}'.format(log_path))

def get_weight(scv_file_path, au_list):
    df = pd.read_csv(scv_file_path)
    au_columns = ['AU'+str(au_name) for au_name in au_list]
    # 过滤无需关注的列
    au_data = df[au_columns].to_numpy()
    
    AUoccur_rate = np.mean(au_data > 0, axis=0)  # 使用 NumPy 计算平均值
    AU_weight = 1.0 / AUoccur_rate
    AU_weight = AU_weight / np.sum(AU_weight) * AU_weight.shape[0]
    
    return torch.tensor(AU_weight)

def set_dataset_info(config):
    if config.exp_name in ["bp4d2disfa"]:
        dataset_info = lambda list: {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU12: {:.2f}'.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4])}
    elif config.exp_name in ["bp4d2gft", "gft2bp4d"]:
        dataset_info = lambda list: {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU23: {:.2f} AU24: {:.2f}'.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9])}        
    elif config.exp_name == "bp4d":
        dataset_info = lambda list: {'AU1: {:.2f} AU2: {:.2f} AU4: {:.2f} AU6: {:.2f} AU7: {:.2f} AU10: {:.2f} AU12: {:.2f} AU14: {:.2f} AU15: {:.2f} AU17: {:.2f} AU23: {:.2f} AU24: {:.2f} '.format(100.*list[0],100.*list[1],100.*list[2],100.*list[3],100.*list[4],100.*list[5],100.*list[6],100.*list[7],100.*list[8],100.*list[9],100.*list[10],100.*list[11])}

    return dataset_info

if __name__ == '__main__':
    
    print(get_weight(edict({
        'source_train_list': '/workspace/projects/uda/Unsupervised-Domain-Adaptation/data/bp4d/train_1f.csv',
        'au_list': [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
    })))