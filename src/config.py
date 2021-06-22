from accelerate import Accelerator
accelerator = Accelerator()

class Config:
    seed = 42,
    model ='swin_small_patch4_window7_224',
    size = 224,
    inp_channels = 1,
    device = accelerator.device,
    lr = 1e-4,
    weight_decay = 1e-6,
    batch_size = 32,
    num_workers = 0,
    epochs = 5,
    out_features = 1,
    name = 'CosineAnnealingLR',
    T_max = 10,
    min_lr = 1e-6,
    num_tta = 1,
    train_dir = '../input/seti-breakthrough-listen/train'
    test_dir = '../input/seti-breakthrough-listen/test'