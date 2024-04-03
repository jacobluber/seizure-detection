import torch
import tifffile
from datetime import datetime
import random
#from Utils.aux import vips2numpy, create_dir
import numpy as np
from sklearn.model_selection import train_test_split 
#import torch
from torchvision import transforms
import os
from os import listdir
from os.path import join
import cv2 as cv
import pickle

class VPdataset():

    def __init__(self,root,split_seed,shuffling_seed,test_ratio,val_ratio
                 ,dataset_type,per_image_normalize=False,transformations=None):
         
         self.root=root
         self.per_image_normalize=per_image_normalize
         self.transformations=transformations
         self.test_ratio=test_ratio
         self.val_ratio=val_ratio
         self.split_seed=split_seed
         self.shuffling_seed=shuffling_seed
         self.dataset_type=dataset_type



         
         



         self.files=[]
         self.train_address=[]
         self.val_address=[]
         self.test_address=[]
            #root='/home/axh5735/projects/compressed_images_HandE/code/with_umap/lung_data'
         
         for file in os.listdir(root):
        
                         path=join(root,file)
                         self.files.append(file)

            
         self.train_files, self.test_files=train_test_split(self.files, 
                                                test_size=self.test_ratio,random_state=self.split_seed, shuffle=True)
         
         self.train_files, self.val_files = train_test_split(self.train_files,
                                                test_size=self.val_ratio,random_state=self.split_seed, shuffle=True)
         

         
         for train_file in self.train_files:
	            
                  for data in os.listdir(join(root,train_file)):
                      
                      if data.endswith(".npy"):
        
                         path=join(root,train_file,data)
                         self.train_address.append(path)

         for val_file in self.val_files:
	            
                  for data in os.listdir(join(root,val_file)):
                      
                      if data.endswith(".npy"):
        
                         path=join(root,val_file,data)
                         self.val_address.append(path)

         for test_file in self.test_files:
	            
                  for data in os.listdir(join(root,test_file)):
                      
                      if data.endswith(".npy"):
        
                         path=join(root,test_file,data)
                         self.test_address.append(path)

         


         random.seed(self.shuffling_seed)
         random.shuffle(self.train_address)
         random.shuffle(self.val_address)
         random.shuffle(self.test_address)

         
         
         
         

            
    

    def _data_to_tensor(self, data):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        output= trans(data)
        output= output.type(torch.FloatTensor)

        return output
         

    def __getitem__(self, index):
     

        
     if self.dataset_type=='train':
        file= self.train_address[index]
        
     elif self.dataset_type=='val':
        file= self.val_address[index]
        
     else:
        file= self.test_address[index]
     
     
     data=np.load(file)

     
     

     data=data/np.max(data)
     


     padding1=np.zeros((7,145))
     #padding1=padding1/10000
     padding2=np.zeros((136,7))
     #padding2=padding2/10000
     

     data=np.concatenate((data,padding1),axis=0)
     data=np.concatenate((data,padding2),axis=1)

     
     
     

     #data= data.astype(np.float)

     

     #print(data.shape)

     output=data.reshape(data.shape[0],data.shape[1],1)
     #output= cv.merge((data,data,data))

     
        
     output=self._data_to_tensor(output)

     

     
     
     
     #print(output.shape)
     if self.per_image_normalize:
            
            std, mean = torch.std_mean(output, dim=(1,2), unbiased=False)
            #print(std.shape)
            #print(mean.shape)
            norm_trans = transforms.Normalize(mean=mean, std=std)
            output = norm_trans(output)

     if self.transformations is not None:
            
            output= self.transformations(output)

     #print(output.shape)

     #print(output.shape)
     
    
     return output, output.size(),file   
     
     

    def __len__(self):
        
        if self.dataset_type=='train':
           return len(self.train_address)
        
        elif self.dataset_type=='val':
        
           return len(self.val_address)
        
        else:
            
           return len(self.test_address)
           

         
         
        
        
         
