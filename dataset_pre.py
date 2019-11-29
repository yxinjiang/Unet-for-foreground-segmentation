
import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import PIL.Image as Image
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A

def walklevel(some_dir, level):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def check_file_exits(file_name):
    if os.path.isfile(file_name):
        return True
    else:
        return False
def resize_img(img,width,height,interpolation = cv2.INTER_CUBIC):
    return cv2.resize(img,(width, height), interpolation = interpolation)

class TrainValDataset(Dataset):
    def __init__(self, train_file,height,width):
        super().__init__()
        
        self.height = height
        self.width = width
        self.data = pd.read_csv(train_file)
        self.file_num = self.data.shape[0]

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        row = self.data.iloc[idx,:]
        in_img_file = row['input_path']
        #bg_img_file = row['bg_path']
        mask_img_file = row['bg_mask_path']

        in_img = resize_img(cv2.imread(in_img_file),self.width, self.height)
        
        #bg_img = resize_img(cv2.imread(bg_img_file),self.width,self.height)
        mask_img = resize_img(cv2.imread(mask_img_file,0),self.width,self.height)[np.newaxis,...]

       
        in_img,mask_img = np.array(in_img),np.array(mask_img)

        in_img = np.transpose(in_img.astype(np.float32) / 255, (2, 0, 1))
        #bg_img = np.transpose(bg_img.astype(np.float32) / 255, (2, 0, 1)) 
        mask_img = mask_img.astype(np.float32) / 255

        sample = {'O': in_img,'M':mask_img}

        return sample



class TestDataset(Dataset):
    def __init__(self,test_file,width,height):
        super().__init__()
        self.height = height
        self.width = width
        self.data = pd.read_csv(test_file)
        self.file_num = self.data.shape[0]

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        row = self.data.iloc[idx,:]
        in_img_file = row['input_path']
        #bg_img_file = row['bg_path']
        mask_img_file = row['bg_mask_path']

        in_img = resize_img(cv2.imread(in_img_file),self.width, self.height)
        
        #bg_img = resize_img(cv2.imread(bg_img_file),self.width,self.height)
        mask_img = resize_img(cv2.imread(mask_img_file,0),self.width,self.height)[np.newaxis,...]

        '''
        in_img = Image.fromarray(in_img)
        bg_img = Image.fromarray(bg_img)
        mask_img = Image.fromarray(mask_img)
        mask_img = mask_img.convert('L')
        '''
        in_img,mask_img = np.array(in_img),np.array(mask_img)

        in_img = np.transpose(in_img.astype(np.float32) / 255, (2, 0, 1))
        #bg_img = np.transpose(bg_img.astype(np.float32) / 255, (2, 0, 1)) 
        mask_img = mask_img.astype(np.float32) / 255

        sample = {'O': in_img,'M':mask_img}

        return sample

def create_dataset_file(root_dir):
    train_data = []
    test_data = []
    for __, dirnames_l0, __ in walklevel(root_dir, level = 0):
        for dirname_l0 in dirnames_l0:
            print ("start dealing with " + dirname_l0)
            dir_10 = os.path.join(root_dir,dirname_l0)
            in_paths = list(paths.list_images(os.path.join(dir_10,"input_model")))
            for in_img_name in in_paths:   
                num = in_img_name.split(os.path.sep)[-1].split('in')[-1]
                bg_img_name = os.path.join(dir_10,'bg_model','bg'+num)
                bg_mask_name = os.path.join(dir_10,'bg_mask','bg_mask'+num)
                if check_file_exits(bg_img_name) and check_file_exits(bg_mask_name):
                    if dirname_l0 != "highway":
                        train_data.append({'input_path':in_img_name,'bg_path':bg_img_name,'bg_mask_path':bg_mask_name})
                    else:
                        test_data.append({'input_path':in_img_name,'bg_path':bg_img_name,'bg_mask_path':bg_mask_name})
    df_train = pd.DataFrame(train_data)      
    df_train.to_csv(os.path.join(root_dir,'train_dataset.csv'))   
    print('create train dataset!')         
    df_test = pd.DataFrame(test_data)      
    df_test.to_csv(os.path.join(root_dir,'test_dataset.csv'))    
    print('create test dataset!')    
     
 

 # helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image,cmap='gray')
    plt.show()
    
# helper function for data visualization    


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

# classes for data loading and preprocessing
class CAMDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            dataset_file, 
            height = 224,
            width = 224,
            classes=['building', 'tree','signsymbol'], 
            augmentation=None, 
            preprocessing=None,
            cropprocessing=None,
    ):
        self.data = pd.read_csv(dataset_file)
        self.file_num = self.data.shape[0]
        self.images_fps = self.data['input_path'].tolist()
        self.masks_fps = self.data['bg_mask_path'].tolist()
        self.height = height
        self.width = width
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float32')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            #background = 1 - mask.sum(axis=-1, keepdims=True)
            background_mask = mask.sum(axis=-1, keepdims=True)
            #mask = np.concatenate((mask, background), axis=-1)
        sample = {'O': image,'M':background_mask}
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(sample)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(sample)
        

        return sample
        
    def __len__(self):
        return self.file_num

def randomcrop(image,output_size,num=3):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    images = []
    h, w = image.shape[:2]
    new_h, new_w = output_size,output_size

    tops = np.random.randint(0, h - new_h,3)
    lefts = np.random.randint(0, w - new_w,3)
    for top,left in zip(tops,lefts):
        image = image[top: top + new_h,left: left + new_w]
        images.append(image)
    return images

class CropPreprocessor(object):

    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
    # store the target image width, height, whether or not
    # horizontal flips should be included, along with the
    # interpolation method used when resizing
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def __call__(self, sample):
		# initialize the list of crops
        image,mask = sample['O'],sample['M']
        h, w = image.shape[:2]     
        new_h, new_w = self.height,self.width     
        tops = np.random.randint(0, h - new_h,size=3)
        tops_end = tops + new_h
        lefts = np.random.randint(0, w - new_w,size=3) 
        lefts_end = lefts + new_w

        crops = []
        mask_crops = []

        # grab the width and height of the image then use these
        # dimensions to define the corners of the image based
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]]

		# compute the center crop of the image as well
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        


        # loop over the coordinates, extract each of the crops,
        # and resize each of them to a fixed size
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height),interpolation=self.inter)
            crops.append(crop)

            mask_crop = mask[startY:endY, startX:endX]
            mask_crop = cv2.resize(mask_crop, (self.width, self.height),
                interpolation=self.inter)
            mask_crops.append(mask_crop)

        for (top,top_end,left,left_end) in zip(tops,tops_end,lefts,lefts_end):
            crop = image[top: top_end,left: left_end]    
            crop = cv2.resize(crop, (self.width, self.height),interpolation=self.inter)
            crops.append(crop)

            mask_crop = mask[top: top_end,left: left_end] 
            mask_crop = cv2.resize(mask_crop, (self.width, self.height),interpolation=self.inter)
            mask_crops.append(mask_crop)
             
        # check to see if the horizontal flips should be taken
        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

            mask_mirrors = [cv2.flip(c, 1) for c in mask_crops]
            mask_crops.extend(mask_mirrors)
        return {'O':crops,'M':mask_crops}

class PreProcessor(object):

    def __init__(self, chanel_first=True):
        self.chanel_first = chanel_first

    def __call__(self, sample):
        image, mask = sample['O'], sample['M']
        

        if self.chanel_first:
            image,mask = np.asarray(image).astype(np.float32),np.asarray(mask).astype(np.float32)
            image = np.transpose(normalize(image), [2, 0, 1])
            mask = np.transpose(normalize(mask[np.newaxis,...]), [2, 0, 1])
        else:
            image = [np.transpose(normalize(c.astype(np.float32)), (2, 0, 1)) for c in image]
            mask = [normalize(c.astype(np.float32)[np.newaxis,...]) for c in mask]

        return {'O': image, 'M': mask}

def get_camvid_dataloader(dataset_type):
        train_file = 'D:\\GIT\\segmentation_models\\data\\CamVid\\camvid_%s_dataset.csv'%dataset_type
        pp = PreProcessor(False)
        cp = CropPreprocessor(224,224)
        dataset = CAMDataset(train_file, classes=['building', 'tree','signsymbol'],augmentation=cp,preprocessing=pp)    
        sample = dataset[0]
        images,masks = sample['O'],sample['M']
        images = np.stack(images,0)
        masks = np.stack(masks,0)


def create_pair_dataset_file(DATA_DIR,images,mask):
    x_dir = os.path.join(DATA_DIR, images)
    y_dir = os.path.join(DATA_DIR, mask)
    ids = os.listdir(x_dir)
    ids = [id.split('.')[0] for id in ids]
    y_ids = os.listdir(y_dir)
    y_ids = [id.split('.')[0] for id in y_ids]
    if set(ids) == set(y_ids):
        images_fps = [os.path.join(x_dir, image_id + '.jpg') for image_id in ids]
        masks_fps = [os.path.join(y_dir, image_id + '.png') for image_id in ids]

        data = pd.DataFrame({'input_path':images_fps,'bg_mask_path':masks_fps})
        data.to_csv('%s\\%s_dataset.csv'%(DATA_DIR,images))
    else:
        print(len(ids))
        print(len(y_ids))
        #print(list(set(ids)-set(y_ids)))
        print('unpaired dataset!')

def create_camvid_dataset_file():
    DATA_DIR = 'D:\\GIT\\segmentation_models\\data\\CamVid\\'

    dataset_types = ['train','val','test']
    for dataset_type in dataset_types:
        create_pair_dataset_file(DATA_DIR,dataset_type,dataset_type+'annot')
        print('create %s dataset'%dataset_type)

    
def create_DUTS_dataset_file():
    DATA_DIR = 'D:\\GIT\\PiCANet-Implementation\\dataset\\DUTS-'

    dataset_types = ['TR','TE']
    for dataset_type in dataset_types:
        create_pair_dataset_file(DATA_DIR+dataset_type,'DUTS-' + dataset_type+'-Image','DUTS-' + dataset_type+'-Mask')
        print('create %s dataset'%dataset_type)
    

if __name__ == "__main__":
    #create_dataset_file('D:\\GIT\\bgsCNN\\dataset')

    #ds = TrainValDataset('D:\\GIT\\bgsCNN\\dataset\\train_dataset.csv',224,224)
    #create_camvid_dataset_file()

    #get_camvid_dataloader('train')
    create_DUTS_dataset_file()
