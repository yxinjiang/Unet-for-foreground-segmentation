import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def normalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    if x_max == x_min:
        x = x/255.0
    else:
        x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x 

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((self.size, self.size), resample=Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                             resample=Image.BILINEAR)
        return {'image': img, 'mask': mask}


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = img.resize((256, 256), resample=Image.BILINEAR), mask.resize((256, 256), resample=Image.BILINEAR)
        h, w = img.size
        new_h, new_w = self.size, self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = img.crop((left, top, left + new_w, top + new_h))
        mask = mask.crop((left, top, left + new_w, top + new_h))

        return {'image': img, 'mask': mask}


class ColorJitter(object):
    def __init__(self, prob):
        self.prob = prob
        self.colorJitter = transforms.ColorJitter(0.1,0.1,0.1)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.colorJitter(img)
            return {'image': img, 'mask': mask}
        else:
            return sample


class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        self.flip = transforms.RandomHorizontalFlip(1.)

    def __call__(self, sample):
        if np.random.random_sample() < self.prob:
            img, mask = sample['image'], sample['mask']
            img = self.flip(img)
            mask = self.flip(mask)
            return {'image': img, 'mask': mask}
        else:
            return sample


class ToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        img, mask = self.tensor(img), self.tensor(mask)
        return {'image': img, 'mask': mask}

#inputs and masks have the same names
class PairDataset(data.Dataset):   
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    def __init__(self, root_dir, images_dir,masks_dir,train=True,camvid=False, data_augmentation=True,classes=['building','tree','signsymbol']):#classes=['building', 'tree','signsymbol']
        self.root_dir = root_dir
        self.train = train
        self.image_list = sorted(os.listdir(os.path.join(root_dir, images_dir)))
        self.mask_list = sorted(os.listdir(os.path.join(root_dir, masks_dir)))
        self.transform = transforms.Compose(
            [RandomFlip(0.5),
             RandomCrop(224),
             ColorJitter(0.5),
             ToTensor()])
        if not (train and data_augmentation):
            self.transform = transforms.Compose([Resize(224), ToTensor()])
        self.root_dir = root_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.data_augmentation = data_augmentation
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.camvid = camvid


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.images_dir, self.image_list[item])
        mask_name = os.path.join(self.root_dir, self.masks_dir, self.mask_list[item])
        img = Image.open(img_name)
        mask = Image.open(mask_name)
        img = img.convert('RGB')
        mask = mask.convert('L')
        if self.camvid:
            masks = [(np.asarray(mask) == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1)
            mask = mask.sum(axis=-1)
            mask = Image.fromarray(np.uint8(mask * 255) , 'L')

        sample = {'image': img, 'mask': mask}

        sample = self.transform(sample)
        return sample


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.image_list = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_name = '{}/{}'.format(self.root_dir, self.image_list[item])
        img = Image.open(img_name)
        sample = img.convert('RGB')
        sample = self.transform(sample)
        return sample