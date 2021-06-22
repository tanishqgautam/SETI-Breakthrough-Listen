# Warning
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Python
import numpy as np
import pandas as pd
import tqdm

# Pytorch for Deep Learning
import torch
from torch.utils.data import DataLoader


from config import Config
from data import SETIDataset
from model import SwinNet
from transforms import Transforms
from utils import prepare_data
from main import best_model_name, best_epoch

train_df = pd.read_csv('../input/seti-breakthrough-listen/train_labels.csv')
test_df = pd.read_csv('../input/seti-breakthrough-listen/sample_submission.csv')
prepare_data(train_df, test_df)

if __name__ == '__main__':
    model = SwinNet()
    model.load_state_dict(torch.load(best_model_name))
    model = model.to(Config.device)

    model.eval()
    predicted_labels = None
    for i in range(Config.num_tta):
        test_dataset = SETIDataset(
            images_filepaths = test_df['image_path'].values,
            targets = test_df['target'].values,
            transform = Transforms.test_transforms
        )
        test_loader = DataLoader(
            test_dataset, batch_size=Config.batch_size,
            shuffle=False, num_workers=Config.num_workers,
            pin_memory=True
        )
        
        temp_preds = None
        with torch.no_grad():
            for (images, target) in tqdm(test_loader):
                images = images.to(Config.device, non_blocking=True)
                output = model(images)
                predictions = torch.sigmoid(output).cpu().numpy()
                if temp_preds is None:
                    temp_preds = predictions
                else:
                    temp_preds = np.vstack((temp_preds, predictions))
        
        if predicted_labels is None:
            predicted_labels = temp_preds
        else:
            predicted_labels += temp_preds
            
    predicted_labels /= Config.num_tta

    torch.save(model.state_dict(), f"{Config.model}_{best_epoch}epochs_weights.pth")

    sub_df = pd.DataFrame()
    sub_df['id'] = test_df['id']
    sub_df['target'] = predicted_labels
    sub_df.to_csv('submission.csv', index=False)