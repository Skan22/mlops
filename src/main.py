# TASK  : Setup the Validation step and epoch 
#       look into early stopping and modelcheckpoiting

#       Contrainerise the entire training scripts and MLFLOW Logging 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import lightning as L
import lightning.pytorch.loggers
import lightning.pytorch.trainer



class LitTinyCNN(L.LightningModule):
        def __init__(self,lr=0.01):
            super().__init__()
            self.save_hyperparameters() # Saves any args given to the initializer

            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(16 * 28 * 28, 10)
            self.loss_fn= nn.CrossEntropyLoss()  # Loss fn lives on model now 

        def forward(self, x):
            return self.fc(self.flatten(self.relu(self.conv1(x))))
        

        def training_step(self,batch, batch_idx):
            x,y_true = batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred,y_true)
            self.log("train loss",loss,prog_bar=True,on_step=True,on_epoch=True,logger=True)
            return loss
        
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            loss = self.loss_fn(self(x), y)
            preds = self(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)


        def configure_optimizers(self):
            return torch.optim.AdamW(params=self.parameters(),lr=self.hparams.lr)
        
        


    # Lightning Data Module 

class MNISTDataModule(L.LightningDataModule):
    def __init__(self,data_dir : str ="./data" , batch_size : int = 64 ):
        super().__init__()
        self.save_hyperparameters()
        self.transform =transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
            )

    def prepare_data(self): # Runs once on rank 0 GPU  , USE IT FOR DOWNLOADING FILES TO DISK ONLY
        torchvision.datasets.MNIST(root=self.hparams.data_dir,train=True,download=True) 
        torchvision.datasets.MNIST(root=self.hparams.data_dir,train=False,download=True) 
            
    def setup(self, stage):
        if stage=="fit":
            self.train_ds =torchvision.datasets.MNIST(root=self.data_dir,train=True,transform=self.transform)
            self.val_ds =torchvision.datasets.MNIST(root=self.data_dir,train=False,transform=self.transform)

    def train_dataloader(self): 
        return torch.utils.data.DataLoader(dataset=self.train_ds,batch_size=self.hparams.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_ds,batch_size=self.hparams.batch_size,num_workers=4,pin_memory=True)
        
        
    # --- 3. MLflow  + Training Loop 
def main():
    mflow_logger = lightning.pytorch.loggers.MLFlowLogger(experiment_name="TinyCNN",
                                                          run_name="local_testing_run_1",
                                                          tracking_uri=os.environ.get("MLFLOW_TRACKING_URI") # picks up the env var in the docker container or local env 
                                                          )
    trainer = lightning.pytorch.trainer.Trainer(max_epochs=2,
                                                logger=mflow_logger,
                                                log_every_n_steps=50,
                                                accelerator="auto"
                                                )
    model = LitTinyCNN()
    dataModule = MNISTDataModule()
    print("Starting Training")
    trainer.fit(model,dataModule)
    print("Training Done")
    
    
if __name__ == "__main__":
    main()
