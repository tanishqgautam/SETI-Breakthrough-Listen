import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from .config import Config

class Transforms:
    train_transforms = albumentations.Compose(
            [
                albumentations.Resize(Config.size, Config.size),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.Rotate(limit=180, p=0.7),
                albumentations.RandomBrightness(limit=0.6, p=0.5),
                albumentations.Cutout(
                    num_holes=10, max_h_size=12, max_w_size=12,
                    fill_value=0, always_apply=False, p=0.5
                ),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.25, scale_limit=0.1, rotate_limit=0
                ),
                ToTensorV2(p=1.0),
            ]
        )

    valid_transforms = albumentations.Compose(
            [
                albumentations.Resize(Config.size, Config.size),
                ToTensorV2(p=1.0)
            ]
        )

    test_transforms = albumentations.Compose(
                [
                    albumentations.Resize(*Config.size, *Config.size),
                    ToTensorV2(p=1.0)
                ]
            )