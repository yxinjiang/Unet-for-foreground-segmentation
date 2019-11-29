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
from dataset_pre import *
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
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log_dir = './logs'
        self.model_path = 'models/latest.pth'
        ensure_dir(self.log_dir)
       
        self.log_name = 'train'
        self.val_log_name = 'val_log'
        self.dataset = 'D:\\GIT\\bgsCNN\\dataset'
        self.test_data_path = os.path.join(self.dataset,'test_dataset.csv')           # test dataset txt file path
        self.train_data_path = os.path.join(self.dataset,'train_dataset.csv')          # train dataset txt file path
        self.cam_dataset = 'D:\\GIT\\segmentation_models\\data\\CamVid'
        self.width = 224
        self.height = 224
        self.net = UNet16().to(self.device)
        print_network(self.net)
        self.step = 0
        self.save_step = 100
        self.batch_size = 16
        self.dataloaders = {}
        self.shuffle = True
        self.opt = Adam(self.net.parameters(),lr=5e-3)
        self.sche = MultiStepLR(self.opt, milestones=[500, 1500,2000,3000], gamma=0.1)
        self.BCE = nn.BCEWithLogitsLoss()
        self.step_time = 0
        self.pretrained = False
        self.train_dataset_len = 0
        self.test_dataset_len = 0
        self.num_classes = 1

    def get_dataloader(self,train_mode=True):            
        if train_mode:
            print('train mode start.......')
            dataset = TrainValDataset(self.train_data_path,self.width,self.height)
            self.train_dataset_len = dataset.__len__()
            self.dataloaders['TrainVal'] = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,  drop_last=True)
            return iter(self.dataloaders['TrainVal'])
        else:
            print('test mode start.......')
            dataset = TestDataset(self.test_data_path,self.width,self.height)
            self.test_dataset_len = dataset.__len__()
            self.dataloaders['Test'] = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)
            return self.dataloaders['Test']
    
    def get_camvid_dataloader(self,dataset_type):
        train_file = self.cam_dataset + '\\camvid_%s_dataset.csv'%dataset_type
        rc = RandomCrop(224)
        pp = PreProcessor()
        dataset = CAMDataset(train_file, classes=['building', 'tree','signsymbol'],augmentation=rc,preprocessing=pp)            
        dataloader = DataLoader(dataset, self.batch_size,self.shuffle)
        if dataset_type == 'train':
            self.train_dataset_len = dataset.__len__()
            return iter(dataloader)
        return dataloader


    def get_camvid_dataset(self,dataset_type):
        train_file = train_file = self.cam_dataset + '\\camvid_%s_dataset.csv'%dataset_type
        pp = PreProcessor(False)
        cp = CropPreprocessor(224,224)
        dataset = CAMDataset(train_file, classes=['building', 'tree','signsymbol'],augmentation=cp,preprocessing=pp)    
        if dataset_type == 'train':
            self.train_dataset_len = dataset.__len__()
        return dataset

    def inf_batch(self, name, batch,from_dataset=False):
        O, M = batch['O'],batch['M']
        if from_dataset == True:
            O, M = torch.from_numpy(np.stack(O,0)),torch.from_numpy(np.stack(M,0))
        O, M = O.to(self.device),M.to(self.device)
        mask_pre= self.net(O)
        
        #if name == 'test':
            #return O.cpu().data,M.cpu().data,mask_pre.cpu().data
        
        # loss
        loss = self.BCE(mask_pre,M)
                
        # log
        
        return  O,M,mask_pre,loss

    

def run_train_val():
    sess = Session()
    dt_train = sess.get_dataloader()
    dt_val = sess.get_dataloader()
    print('train dataset len ',sess.train_dataset_len)
    if sess.pretrained:
        sess.net.load_state_dict(torch.load(sess.model_path))
    while sess.step < sess.train_dataset_len//sess.batch_size:
        start = timeit.default_timer()
        sess.sche.step()
        sess.net.train()        
        sess.net.zero_grad()

        batch_t = next(dt_train)
        in_img_t,bg_img_t,mask_true_t, mask_pre_t,loss_t = sess.inf_batch('train', batch_t)
        
        loss_t.backward()
        sess.step_time = timeit.default_timer() - start
        
        sess.opt.step() 
        

        if sess.step % 100 == 0:
            
            sess.net.eval()
            batch_v = next(dt_val)
            in_img_v,bg_img_v,mask_true_v, mask_pre_v,loss_v = sess.inf_batch('val', batch_v)
            
            torchvision.utils.save_image(in_img_v,'./logs/val_in_img_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_true_v,'./logs/val_mask_true_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_pre_v,'./logs/val_mask_pre_v %d.png'%sess.step)
            
            print('%d step loss: %.4f, train step time: %.2f'%(sess.step,loss_v,sess.step_time))
            torch.save(sess.net.state_dict(), sess.model_path)
        sess.step += 1

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
            img = np.asarray(cv2.resize(img,(sess.width,sess.height))).astype('float32')
            cv2.imwrite('test/'+fname+'_input.png',img)
            img = np.transpose(normalize(img),(2,0,1))
            img = np.expand_dims(img,0)
            img = torch.from_numpy(img)
            mask_pre = sess.net(img.to(sess.device))
            mask_pre = mask_pre.cpu().detach().numpy().squeeze()*255
            mask_pre = mask_pre.clip(0,255)
            #mask_pre = np.transpose(mask_pre,(1,2,0))*255
            cv2.imwrite('test/'+fname+'_mask_pre.png',mask_pre)

def run_train_camvid_dataloader(sess):
    
    dt_train = sess.get_camvid_dataloader('train')
    dt_val = sess.get_camvid_dataloader('val')
    print('train dataset len ',sess.train_dataset_len)
    while sess.step < sess.train_dataset_len:
        start = timeit.default_timer()
        sess.sche.step()
        sess.net.train()
        sess.net.zero_grad()
        batch_t = next(dt_train)
        in_img_t,mask_true_t, mask_pre_t,loss_t = sess.inf_batch('train', batch_t,True)
        
        loss_t.backward()
        sess.step_time = timeit.default_timer() - start
        
        sess.opt.step() 
        

        if sess.step % 4 == 0:
            
            sess.net.eval()
            batch_v = next(dt_val)
            in_img_v,mask_true_v, mask_pre_v,loss_v = sess.inf_batch('val', batch_v,True)
            
            torchvision.utils.save_image(in_img_v,'./logs/camvid_val_in_img_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_true_v,'./logs/camvid_val_mask_true_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_pre_v,'./logs/camvid_val_mask_pre_v %d.png'%sess.step)
            
            print('%d step loss: %.4f, train step time: %.2f'%(sess.step,loss_v,sess.step_time))
            torch.save(sess.net.state_dict(), sess.model_path)
        sess.step += 1


def run_train_camvid_dataset(sess):
    
    if sess.pretrained:
        sess.net.load_state_dict(torch.load(sess.model_path))
        print('load pretrained model..........')
    start = timeit.default_timer()
    dt_train = sess.get_camvid_dataset('train')
    dt_val = sess.get_camvid_dataset('test')
    print('train dataset len ',sess.train_dataset_len)
    while sess.step < sess.train_dataset_len:
        sess.sche.step()
        sess.net.train()
        sess.net.zero_grad()
        batch_t = dt_train[sess.step]
        in_img_t,mask_true_t, mask_pre_t,loss_t = sess.inf_batch('train', batch_t,True)
        loss_t.backward()
        sess.step_time = timeit.default_timer() - start
        sess.opt.step() 
        
        if sess.step % 4 == 0:            
            sess.net.eval()
            batch_v = dt_val[sess.step//4]
            in_img_v,mask_true_v, mask_pre_v,loss_v = sess.inf_batch('test', batch_v,True)
            
            torchvision.utils.save_image(in_img_v,'./logs/camvid_val_in_img_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_true_v,'./logs/camvid_val_mask_true_v %d.png'%sess.step)
            torchvision.utils.save_image(mask_pre_v,'./logs/camvid_val_mask_pre_v %d.png'%sess.step)
            
            print('%d step loss: %.4f, train step time: %.2f'%(sess.step,loss_v,sess.step_time))
            torch.save(sess.net.state_dict(), sess.model_path)
        sess.step += 1



if __name__ == "__main__":
   #run_train_val() 
   #run_test()
   sess = Session()
   #sess.pretrained = True
   #sess.batch_size = 4
   #run_train_camvid_dataset(sess)
   img_paths = list(paths.list_images('C:\\Users\\yixin\\Pictures\\QQplayerPic'))
   test_customer_img_list(sess,img_paths)
   #run_train_camvid_dataloader(sess)
