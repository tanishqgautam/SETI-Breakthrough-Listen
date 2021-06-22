from tqdm import tqdm
from torch.cuda import amp
from accelerate import Accelerator
import wandb

accelerator = Accelerator()

from .utils import MetricMonitor, cutmix, cutmix_criterion, use_roc_score
from .config import Config

def train(train_loader, model, criterion, optimizer, epoch):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    scaler = amp.GradScaler()
       
    for i, (images, target) in enumerate(stream, start=1):

        images = images.to(Config.device)
        target = target.to(Config.device).float().view(-1, 1)
        images, targets_a, targets_b, lam = cutmix(images, target.view(-1, 1))
        
        with amp.autocast(enabled=True):
            output = model(images)
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)
            
        accelerator.backward(scaler.scale(loss))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        roc_score = use_roc_score(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('ROC', roc_score)
        wandb.log({"Train Epoch":epoch,"Train loss": loss.item(), "Train ROC":roc_score})
        

        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch,
                metric_monitor=metric_monitor)
        )