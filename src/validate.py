import torch
from tqdm import tqdm
import wandb

from .utils import MetricMonitor, use_roc_score
from .config import Config

def validate(val_loader, model, criterion, epoch):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True).float().view(-1, 1)
            output = model(images)
            loss = criterion(output, target)
            roc_score = use_roc_score(output, target)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('ROC', roc_score)
            wandb.log({"Valid Epoch": epoch, "Valid loss": loss.item(), "Valid ROC":roc_score})
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
            
            targets = target.detach().cpu().numpy().tolist()
            outputs = output.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
    return final_outputs, final_targets