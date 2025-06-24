import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import csv
import pdb


class PseudoLabelDataset(Dataset):
    
    def __init__(self, images_list, truth_labels, pseudo_labels, config=None):
        
        self.config = config
        self.images_list = images_list
        self.truth_labels = truth_labels
        self.pseudo_labels = pseudo_labels
        normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]) # CLIP
        self.transforms = T.Compose([
            T.RandomResizedCrop((224, 224), (0.9, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self,index):
        img_name = self.images_list[index]
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)
        truth_label = self.truth_labels[index]
        pseudo_label = self.pseudo_labels[index]
        return img, truth_label, pseudo_label, img_name

class GeneralData(Dataset):
    def __init__(self, root, data_list_file, config, phase = "train"):
        self.config = config
        self.input_shape = config.input_shape
        self.au_list = config.au_list
        self.phase = phase
        self.root = root
        self.image_list, self.label_list = self.read_file(root, data_list_file)

        # normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]) # CLIP
        
        # 不同的阶段使用不同的数据增强策略
        if self.phase == 'train':
            self.transforms = T.Compose([
                # T.Resize((self.input_shape[1] + 24,self.input_shape[1] + 24)),
                # T.RandomRotation(10),
                # T.RandomResizedCrop((self.input_shape[1], self.input_shape[2]), (0.9, 1.0)),
                T.RandomCrop(self.input_shape[1]),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.config.img_size), # 256
                T.CenterCrop(self.input_shape[1]), # 224
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,index):
        img_name = self.image_list[index]
        img = Image.open(img_name).convert('RGB')
        img = self.transforms(img)

        current_lbl = np.array(self.label_list[index])
        current_lbl = torch.from_numpy(current_lbl)

        return img, current_lbl, img_name

    def read_file(self, prefix, file_name):
        img_list = []
        lbl_list = []
        aus = ['AU'+str(au_name) for au_name in self.au_list]
        with open(file_name, 'r') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                # row: {'img_path': 'BP4D/Images/2F23_10/1001.jpg', 'person': '2F23', 'AU1': '1', 'AU2': '1', 'AU4': '0', 'AU6': '0', 'AU7': '0', 'AU10': '0', 'AU12': '0', 'AU14': '0', 'AU15': '0', 'AU17': '0', 'AU23': '0', 'AU24': '0'}
                img_name = os.path.join(prefix, row['img_path'])
                lbl = [int(row[au]) for au in aus]

                img_list.append(img_name)
                lbl_list.append(lbl)
        
        return img_list, lbl_list


def make_dataset(image_list, label_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :], au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class BP4D(Dataset):
    def __init__(self, root_path, data_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path # 数据list文件目录
        self._data_path = data_path # 图片文件所在目录
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        # 图片文件所在目录
        self.img_folder_path = os.path.join(data_path)
        if self._train:
            # img
            # pdb.set_trace()
            train_image_list_path = os.path.join(root_path, 'list', 'BP4D_train_img_path_fold' + str(fold) +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'BP4D_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'BP4D_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'BP4D_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'BP4D_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)
            
        # pdb.set_trace()

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            # 'BP4D/Images/2F23_10/1001.jpg'
            # array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, data_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path # 数据list文件目录
        self._data_path = data_path # 图片文件所在目录
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(data_path)
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'DISFA_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)


class GFT(Dataset):
    def __init__(self, root_path, data_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path # 数据list文件目录
        self._data_path = data_path # 图片文件所在目录
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(data_path)
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'GFT_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'GFT_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

            # AU relation
            if self._stage == 2:
                au_relation_list_path = os.path.join(root_path, 'list', 'GFT_train_AU_relation_fold' + str(fold) + '.txt')
                au_relation_list = np.loadtxt(au_relation_list_path)
                self.data_list = make_dataset(train_image_list, train_label_list, au_relation_list)
            else:
                self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'GFT_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'GFT_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label, au_relation = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
            return img, label, au_relation
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path,img))

            if self._train:
                w, h = img.size
                offset_y = random.randint(0, h - self.crop_size)
                offset_x = random.randint(0, w - self.crop_size)
                flip = random.randint(0, 1)
                if self._transform is not None:
                    img = self._transform(img, flip, offset_x, offset_y)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)