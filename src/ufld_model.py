"""Kien truc UFLD (Ultra-Fast-Lane-Detection, Qin et al. 2020), rut gon tu repo goc
(cfzd/Ultra-Fast-Lane-Detection) chi giu phan can cho INFERENCE (bo nhanh aux segmentation
vi no chi dung luc training). Phai giu dung ten layer/shape thi moi load duoc checkpoint
pretrained (state_dict khop theo ten key)."""
import numpy as np
import torch
import torchvision


class ResNetBackbone(torch.nn.Module):
    """Boc lai resnet18 cua torchvision, lay ra 3 tang feature map trung gian (x2, x3, x4)
    giong cach UFLD goc trich xuat feature de dua vao cls head."""

    def __init__(self, pretrained=False):
        super().__init__()
        model = torchvision.models.resnet18(weights=None if not pretrained else "DEFAULT")
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class ParsingNet(torch.nn.Module):
    """cls_dim = (griding_num+1, so_row_anchor, so_lane). Voi config CULane goc:
    (201, 18, 4) - 200 grid cell + 1 lop "khong co lane", 18 row anchor, 4 lane."""

    def __init__(self, cls_dim=(201, 18, 4)):
        super().__init__()
        self.cls_dim = cls_dim
        self.total_dim = int(np.prod(cls_dim))
        self.model = ResNetBackbone(pretrained=False)
        self.pool = torch.nn.Conv2d(512, 8, 1)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

    def forward(self, x):
        _, _, fea = self.model(x)
        fea = self.pool(fea).view(-1, 1800)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)
        return group_cls


def load_pretrained(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)["model"]
    # checkpoint train bang DataParallel nen key co tien to "module." can bo di
    compatible = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(compatible, strict=False)  # strict=False: bo qua aux_* khong dung khi infer
    return model
