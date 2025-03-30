# python native
import os
import json
import random

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler,RandomSampler,SequentialSampler
from utils import set_seed

# seed 666
set_seed()

def resize_img(img, label, s, is_train):
    if is_train:
        img = cv2.resize(img, s, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, s, interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_LINEAR)
        label = label
    return img, label


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None,img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 512
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size   


class XRayDataset(Dataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, gray=False):
        
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        jsons = {
            os.path.relpath(os.path.join(root, fname), start=label_root)
            for root, _dirs, files in os.walk(label_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".json"
        }
        
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
        pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

        assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
        assert len(pngs_fn_prefix - jsons_fn_prefix) == 0
        
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 20% valid
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
            
        df = pd.read_csv('/data/ephemeral/home/data/meta_data.csv')
        df = df.iloc[:550, :5]
        df.rename(columns={'나이' : 'age', '키(신장)': 'height', '체중(몸무게)' : 'weight'}, inplace=True)
        df['ID'] = ['ID{0:03d}'.format(int(id)) for id in df['ID']]
        height_bins = [140, 160, 170, 190]
        df['height_category'] = pd.cut(df['height'], bins=height_bins, labels=False)
        
        self.image_root = image_root
        self.label_root = label_root
        self.classes = [
            'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
            'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
            'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
            'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
            'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
            'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
        ]
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.gray = gray
        self.meta = {df['ID'][i]:df['height_category'][i] for i in range(len(df))}
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):

        CLASS2IND = {v: i for i, v in enumerate(self.classes)}
        IND2CLASS = {v: k for k, v in CLASS2IND.items()}
        
        image_name = self.filenames[item]
        image_id = image_name.split('/')[0]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.classes), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        # to tenser will be done later
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        aux_label = self.meta[image_id]
            
        return image, label, image_path, aux_label
    

class XRayMultiScaleDataset(XRayDataset):
    def __init__(self, image_root, label_root, is_train=True, transforms=None, gray=False):
        super(XRayMultiScaleDataset, self).__init__(image_root, label_root, is_train, transforms, gray)
    
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            item, input_size = item
        else:
            # set the default image size here
            input_size = 512

        CLASS2IND = {v: i for i, v in enumerate(self.classes)}
        IND2CLASS = {v: k for k, v in CLASS2IND.items()}
        
        image_name = self.filenames[item]
        image_id = image_name.split('/')[0]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.
        
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_root, label_name)
        
        # (H, W, NC) 모양의 label을 생성합니다.
        label_shape = tuple(image.shape[:2]) + (len(self.classes), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # label 파일을 읽습니다.
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # 클래스 별로 처리합니다.
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon 포맷을 dense한 mask 포맷으로 바꿉니다.
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
            
        image, label = resize_img(image, label, (input_size, input_size), self.is_train)
        
        if self.gray:
            image = image[..., np.newaxis]
        
        # to tenser will be done later
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        aux_label = self.meta[image_id]
            
        return image, label, image_path, aux_label
    
    
    
class XRayInferenceDataset(Dataset):
    def __init__(self, image_root, transforms=None, gray=False):
        pngs = {
            os.path.relpath(os.path.join(root, fname), start=image_root)
            for root, _dirs, files in os.walk(image_root)
            for fname in files
            if os.path.splitext(fname)[1].lower() == ".png"
        }
        
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))
        
        self.image_root = image_root
        self.filenames = _filenames
        self.transforms = transforms
        self.gray = gray
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.
        
        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        if self.gray:
            image = image[..., np.newaxis]
        # to tenser will be done later
        image = image.transpose(2, 0, 1)  
        
        image = torch.from_numpy(image).float()
            
        return image, image_name
