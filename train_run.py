import os
import sys
import cv2
import argparse
import numpy as np
import logging

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import skimage.measure as ms
import progressbar
import torchvision.utils
from dataset import * 
import timeit
from unet_models import *
from eval import eval_net
from imutils import paths
from PIL import Image

torch.cuda.manual_seed_all(2019)
torch.manual_seed(2019)

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



class Session:
    def __init__(self,dataset_dir = './dataset',images_dir='images',masks_dir = 'masks',train=True,camvid=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log_dir = './logs'
        self.model_path = './models/latest_foreground.pth'
        ensure_dir(self.log_dir)
        self.dataset_dir = dataset_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.width = 224
        self.height = 224
        self.net = UNet16().to(self.device)
        self.train = train
        self.data_augmentation = True
        self.camvid = camvid
        self.step = 0
        self.save_step = 4
        self.epochs = 16
        self.batch_size = 16
        self.opt = Adam(self.net.parameters(),lr=5e-3)
        self.sche = MultiStepLR(self.opt, milestones=[500, 1500,2000,3000], gamma=0.1)
        self.BCE = nn.BCEWithLogitsLoss()
        self.step_time = 0
        self.pretrained = True
        self.dataset_len = 0
        self.num_classes = 1

    def get_dataloader(self,train=True): 
        dataset = PairDataset(self.dataset_dir,self.images_dir,self.masks_dir,self.train,self.data_augmentation,self.camvid)
        self.dataset_len = dataset.__len__()
        if train:             
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return iter(dataloader)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            return dataloader


    def inf_batch(self, name, batch):
        images, masks = batch['image'],batch['mask']
        images, masks = images.to(self.device),masks.to(self.device)
        mask_pre= self.net(images)
        # loss
        loss = self.BCE(mask_pre,masks)
        
        return  mask_pre,loss

    def run_train_val(self):
        if self.pretrained:
            print('load pretrained model........')
            self.net.load_state_dict(torch.load(self.model_path))
        for epoch in range(self.epochs):
            print('%d epoch begin......'%(epoch))
            dt_train = self.get_dataloader()
            dt_val = self.get_dataloader()
            self.step = 0
            while self.step < self.dataset_len//self.batch_size:
                start = timeit.default_timer()
                self.sche.step()
                self.net.train()        
                self.net.zero_grad()
                batch_tr = next(dt_train)
                mask_pre_t,loss_t = self.inf_batch('train', batch_tr)                
                loss_t.backward()
                self.step_time = timeit.default_timer() - start                
                self.opt.step() 
                if self.step % self.save_step == 0:            
                    self.net.eval()
                    batch_v = next(dt_val)
                    mask_pre_v,loss_v = self.inf_batch('val', batch_v)                    
                    torchvision.utils.save_image(batch_v['image'],'./logs/epoch %d_val_img_step_%d.png'%(epoch,self.step))
                    torchvision.utils.save_image(batch_v['mask'],'./logs/epoch %d_val_mask_true_step_%d.png'%(epoch,self.step))
                    torchvision.utils.save_image(mask_pre_v,'./logs/epoch %d_val_mask_pre_step_%d.png'%(epoch,self.step))                   
                    print('%d step loss: %.4f, train step time: %.2f'%(self.step,loss_v,self.step_time))
                    #torch.save(self.net.state_dict(), self.model_path)
                self.step += 1
            if epoch %4 == 0:
                torch.save(self.net.state_dict(), self.model_path)

def run_test():
    sess = Session()
    #sess.net = sess.net.load_state_dict(torch.load(sess.model_path))
    sess.batch_size = 1
    sess.shuffle = False
    dt = sess.get_dataloader(train_mode=False)   
    val_score = eval_net(sess.net, dt, sess.device, sess.test_dataset_len)
    if sess.net.num_classes > 1:
        print('Validation cross entropy: {}'.format(val_score))

    else:
        print('Validation Dice Coeff: {}'.format(val_score))

def test_customer_img_list(sess,img_paths=[]):
    sess.net.load_state_dict(torch.load(sess.model_path))
    sess.net.eval()
    if len(img_paths):
        for pth in img_paths:
            fname = pth.split(os.path.sep)[-1].split('.')[0]
            img = cv2.imread(pth)
            frame = img.copy()
            width,height,_ = img.shape
            img = np.asarray(cv2.resize(img,(sess.width,sess.height))).astype('float32')            
            cv2.imwrite('test results/'+fname+'_input.png',frame)
            img = np.transpose(normalize(img),(2,0,1))
            img = np.expand_dims(img,0)
            img = torch.from_numpy(img)
            mask_pre = sess.net(img.to(sess.device))
            mask_pre = mask_pre.cpu().detach().numpy().squeeze()
            mask_pre = mask_pre*255
            mask_pre = mask_pre.clip(0,255)
            #cv2.imwrite('test/'+fname+'_mask_pre.png',mask_pre)
            mask_pre_resize = cv2.resize(mask_pre,(height,width),cv2.INTER_NEAREST)
            cv2.imwrite('test results/'+fname+'_mask_pre.png',mask_pre_resize)
            ind = mask_pre_resize[:,:]>0
            ind = np.dstack((ind, ind, ind))
            cv2.imwrite('test results/'+fname+'_input_with_mask_INTER_NEAREST.png',np.multiply(frame,ind))
            mask_processed = find_max_contour(mask_pre_resize)
            cv2.imwrite('test results/'+fname+'_mask_processed.png',mask_processed)
        print('total %d images processed....'%len(img_paths))


def find_max_contour(image):
    image = image.astype('uint8')
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(labels.shape)
    img2[labels == max_label] = 255
    return img2

if __name__ == "__main__":
    '''
    dataset_dir = 'D:\\GIT\\PiCANet-Implementation\\dataset\\DUTS-TR'
    images_dir = 'DUTS-TR-Image'
    masks_dir = 'DUTS-TR-Mask'
    sess = Session(dataset_dir,images_dir,masks_dir)
    run_train_val(sess)
    '''
    dataset_dir = 'D:\\GIT\\00Dataset\\Segmentation\\camvid'
    images_dir = 'images'
    masks_dir = 'masks'
    sess = Session(dataset_dir,images_dir,masks_dir,True,True)
    sess.pretrained=False
    sess.save_step = 40
    sess.epochs = 16
    #sess.run_train_val()
    img_paths = list(paths.list_images('C:\\Users\\yixin\\Pictures\\QQplayerPic'))
    test_customer_img_list(sess,img_paths)
   #run_train_camvid_dataloader(sess)
