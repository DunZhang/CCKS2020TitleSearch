import torch
import os


class ManualFeatureBasedAttrScoerModel(torch.nn.Module):
    def __init__(self, ):
        super(ManualFeatureBasedAttrScoerModel, self).__init__()

    def forward(self, features):
        return torch.sum(features, dim=1, keepdim=True)

    def save(self, save_dir):
        if hasattr(self, "fc"):
            torch.save(self.fc.state_dict(), os.path.join(save_dir, "fc_weight.bin"))
