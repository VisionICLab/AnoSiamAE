import os
import torch
import json
from networks.data.build import build_dataloader

from networks.config.config import save_cfg
from networks.engine.base import _BaseModel


class Trainer():
    def __init__(self, model:_BaseModel, cfg:dict, device:str = "cuda", test_only:bool= False):
        # Configuration parameters 
        self.cfg = cfg
        self.device = device
        self.seed = self.cfg["SYSTEM"]["SEED"]
        self.test_only = test_only
        self.model = model

        self.dataloader = build_dataloader(cfg)
        
        # Optimization configuration
        self.batch_size = self.cfg["DATA"]["BATCH_SIZE"]
        self.epochs = self.cfg["DATA"]["EPOCHS"]
        self.start_epoch = self.model.start_epoch

        # Saving configuration
        self.model_output = os.path.join(self.cfg["OUTPUT"],"model")
        if not os.path.exists(self.model_output):
            os.makedirs(self.model_output)
        self.log_output = os.path.join(self.cfg["OUTPUT"],"log")
        if not os.path.exists(self.log_output):
            os.makedirs(self.log_output)
        self.train_output = os.path.join(self.cfg["OUTPUT"],"train",)
        if not os.path.exists(self.train_output):
            os.makedirs(self.train_output)
        self.val_output = os.path.join(self.cfg["OUTPUT"],"validation",)
        if not os.path.exists(self.val_output):
            os.makedirs(self.val_output)
        self.test_output = os.path.join(self.cfg["OUTPUT"],"test",)
        if not os.path.exists(self.test_output):
            os.makedirs(self.test_output)
        save_cfg(self.cfg, self.cfg["OUTPUT"])


    def run(self):
        if not self.test_only:
            self.train()
        self.test()
        

    def train(self):       
        #Starting Training

        # Initialization best checkpoint parameters
        print("Beginning of the training process...")
        for epoch in range(self.start_epoch, self.epochs):

            # One epoch training
            train_dict, save_dict = self.model.train_model(epoch=epoch, 
                                                           train_data=self.dataloader.train_dataloader(), 
                                                           output_dir=self.train_output, 
                                                           visualization = True, 
                                                           save_frq = self.cfg["EVALUATION"]["TRAIN_FRQ"])
            print("TRAINING: [Epoch {}/{}] || {}".format(epoch, self.epochs, train_dict))
            torch.save(save_dict, os.path.join(self.model_output,'checkpoint.pth'))

            # Saving training logs
            with open(os.path.join(self.log_output, "train_log.txt"), "a") as f:
                    f.write(json.dumps(train_dict) +'\n')

            # One epoch validation
            if epoch==self.start_epoch or epoch%self.cfg["EVALUATION"]["VAL_FRQ"] ==0 or epoch==self.epochs-1:
                val_dict = self.model.eval_model(epoch=epoch, 
                                                 val_data=self.dataloader.val_dataloader(), 
                                                 Test=False,
                                                 output_dir=self.val_output)
                
                # Saving validation logs
                with open(os.path.join(self.log_output, "val_log.txt"), "a") as f:
                    f.write(json.dumps(val_dict) + '\n')

            # Save current checkpoint
            torch.save(save_dict, os.path.join(self.model_output, 'checkpoint.pth'))
        print("... End of the training process")
        
    def test(self):
        # One epoch testing
        print("Beginning of the testing process...")
        test_dict = self.model.eval_model(epoch=self.epochs, 
                                          val_data=self.dataloader.test_dataloader(), 
                                          Test=True,
                                          output_dir=self.val_output)
        # Saving testing logs
        with open(os.path.join(self.log_output, "test_log.txt"), "a") as f:
            f.write(json.dumps(test_dict) + '\n')
        
        print("... End of the testing process")

    
 

        



