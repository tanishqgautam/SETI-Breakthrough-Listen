# Warning
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Python
import pandas as pd
import numpy as np

# Utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Pytorch for Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# GPU 
from accelerate import Accelerator
accelerator = Accelerator()

# Weights and Biases Tool
import wandb

from config import Config
from data import SETIDataset
from model import SwinNet
from train import train
from validate import validate
from transforms import Transforms
from utils import seed_everything, prepare_data, get_sampler

seed_everything(Config.seed)

train_df = pd.read_csv('../input/seti-breakthrough-listen/train_labels.csv')
test_df = pd.read_csv('../input/seti-breakthrough-listen/sample_submission.csv')
prepare_data(train_df, test_df)


(X_train, X_valid, y_train, y_valid) = train_test_split(train_df['image_path'],
                                                        train_df['target'],
                                                        test_size=0.2,
                                                        stratify=train_df['target'],
                                                        shuffle=True,
                                                        random_state=Config.seed)

train_dataset = SETIDataset(
    images_filepaths=X_train.values,
    targets=y_train.values,
    transform=Transforms.train_transforms
)

valid_dataset = SETIDataset(
    images_filepaths=X_valid.values,
    targets=y_valid.values,
    transform=Transforms.valid_transforms
)


train_loader = DataLoader(
    train_dataset, batch_size=Config.batch_size, sampler = get_sampler(y_train),
    num_workers=Config.num_workers, pin_memory=True)

val_loader = DataLoader(
    valid_dataset, batch_size=Config.batch_size, shuffle=False,
    num_workers=Config.num_workers, pin_memory=True)

model = SwinNet()
model = model.to(Config.device)
criterion = nn.BCEWithLogitsLoss().to(Config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr,
                             weight_decay=Config.weight_decay,
                             amsgrad=False)

scheduler = CosineAnnealingLR(optimizer,
                              T_max=Config.T_max,
                              eta_min=Config.min_lr,
                              last_epoch=-1)


if __name__ == '__main__':
    best_roc = -np.inf
    best_epoch = -np.inf
    best_model_name = None

    for epoch in range(1, Config.epochs + 1):
        
        run = wandb.init(project='Seti-Swin', 
                    config=Config, 
                    job_type='train',
                    name = 'Swin Transformer')
        
        train(train_loader, model, criterion, optimizer, epoch)
        predictions, valid_targets = validate(val_loader, model, criterion, epoch)
        roc_auc = round(roc_auc_score(valid_targets, predictions), 3)
        torch.save(model.state_dict(),f"{Config.model}_{epoch}_epoch_{roc_auc}_roc_auc.pth")
        
        if roc_auc > best_roc:
            best_roc = roc_auc
            best_epoch = epoch
            best_model_name = f"{Config.model}_{epoch}_epoch_{roc_auc}_roc_auc.pth"
            
        scheduler.step()            

    print(f'The best ROC: {best_roc} was achieved on epoch: {best_epoch}.')
    print(f'The Best saved model is: {best_model_name}')                   