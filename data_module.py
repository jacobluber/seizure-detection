import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
import numpy as np
import argparse
from os import makedirs, path
from argparse import ArgumentParser
from util import calculating_stat
from dataset import VPdataset
from Utils.aux import create_dir, save_transformation, load_transformation
from pytorch_lightning.trainer.states import TrainerFn


class VP_DataModule(pl.LightningDataModule):

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        

        # -> Datasset Args

        parser.add_argument(
            "--root",
            type = str,
            default = '/home/axh5735/projects/signal_compression/data_sep',
            help = "Address of the dataset directory."
        )
        
        
        
        parser.add_argument(
            "--transformations_write_dir",
            type = str,
            default = None,
            help = "Directory defining where to save the generated transformations and inverse transformations .obj files. If not provided, all generated coordinate files will be stored in './logs/tb_logs/logging_name/'. [default: None]"
        )

        

        
        
        
        parser.add_argument(
            "--test_ratio",
            type = float,
            default = 0.1,
            help = ""
        )
        
        
        parser.add_argument(
            "--val_ratio",
            type = float,
            default = 0.1,
            help = ""
        )
        
        
        parser.add_argument(
            "--split_seed",
            type = int,
            default = 2,
            help = ""
        )

        parser.add_argument(
            "--shuffling_seed",
            type = int,
            default = 2,
            help = ""
        )
        
       

        parser.add_argument(
            "--per_image_normalize",
            action = argparse.BooleanOptionalAction,
            help = "Whether to normalize each patch with respect to itself."
        )


        parser.add_argument(
            "--prepare",
            action = argparse.BooleanOptionalAction,
            help = "getting coords."
        )

        
        

        # -> Data Module Args

        parser.add_argument(
            "--batch_size",
            type = int,
            default = 128,
            help = "The batch size used with all dataloaders. [default: 128]"
        )

        parser.add_argument(
            "--num_dataloader_workers",
            type = int,
            default = 8,
            help = "Number of processor workers used for dataloaders. [default: 8]"
        )

    
        parser.add_argument(
            "--normalize_transform",
            action = argparse.BooleanOptionalAction,
            help = "If passed, DataModule will calculate or load the whole training dataset mean and std per channel and passes it to transforms."
        )

        
        
    

        

        parser.add_argument(
            "--transformations_read_dir",
            type = str,
            default = None,
            help = "Directory defining where to write the coords'. [default: None]"
        )


        return parser



    def __init__(
        self,
        root,
        transformations_write_dir,
        batch_size,
        num_dataloader_workers,
        per_image_normalize,
        normalize_transform,
        test_ratio,
        val_ratio,
        split_seed,
        shuffling_seed,
        prepare,
        transformations_read_dir,
        *args,
        **kwargs,
                                ):
       

        super().__init__()
        
        self.root=root
        self.transformations_write_dir=transformations_write_dir
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        self.per_image_normalize = per_image_normalize
        self.normalize_transform = normalize_transform
        self.test_ratio=test_ratio
        self.val_ratio=val_ratio
        self.split_seed=split_seed
        self.shuffling_seed=shuffling_seed
        self.prepare=prepare
        self.transformations_read_dir=transformations_read_dir

        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.dataset_kwargs = {

            "root":self.root,
            "test_ratio":self.test_ratio,
            "val_ratio":self.val_ratio,
            "split_seed":self.split_seed,
            "shuffling_seed": self.shuffling_seed,
            "per_image_normalize": per_image_normalize,
            
        }


        

        
        
        
    def prepare_data(self):
        
        if self.prepare:
           
           train_dataset = VPdataset (dataset_type="train",transformations=None, **self.dataset_kwargs)
                
        # All stats should be calculated at highest stable batch_size to reduce approximation errors for mean and std

           loader = DataLoader(train_dataset, batch_size=256,num_workers=self.num_dataloader_workers)
           calculating_stat(loader)


    def setup(self, stage=None):
        

        if self.prepare:
        # Determining transformations to apply.

        
              transforms_list = []
              inverse_transforms_list = []

              if self.normalize_transform:
            
                  std = np.loadtxt(path.join("/home/axh5735/projects/signal_compression/logs/compression_128_256/stat", "std.gz"))
                  mean = np.loadtxt(path.join("/home/axh5735/projects/signal_compression/logs/compression_128_256/stat", "mean.gz"))

            

                  transforms_list.append(
                      transforms.Normalize(mean=mean, std=std)
                  )

                  inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1])))
                  inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0]), std=1/std))

            

              transformations = transforms.Compose(transforms_list)
              inverse_transformations = transforms.Compose(inverse_transforms_list)

              # Saving transformations to file
              save_transformation(transformations, path.join(self.transformations_write_dir, "trans.obj"))
              save_transformation(inverse_transformations, path.join(self.transformations_write_dir, "inv_trans.obj"))


        # Creating corresponding datasets

        transformations= load_transformation(path.join(self.transformations_read_dir, "trans.obj"))

        if stage in (None, "fit"):
            self.train_dataset = VPdataset(dataset_type="train", transformations=transformations,
                                           **self.dataset_kwargs)
          
            self.val_dataset = VPdataset(dataset_type="val", transformations=transformations,
                                        **self.dataset_kwargs)
            
            


        elif stage in (None, "validate"):
            self.val_dataset = VPdataset(dataset_type="val", transformations=transformations,
                                         **self.dataset_kwargs)
           

        elif stage in (None, "test"):
           
            self.test_dataset = VPdataset(dataset_type="test", transformations=transformations,
                                          **self.dataset_kwargs)
            

       
        
    def train_dataloader(self):

        
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)
    
    
    def val_dataloader(self):

        
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)

    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
        num_workers=self.num_dataloader_workers)    
