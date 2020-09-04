import torch
import os


class ManualFeatureBasedAttrScoerModel(torch.nn.Module):
    def __init__(self, num_features, fc_path=None):
        super(ManualFeatureBasedAttrScoerModel, self).__init__()
        self.fc = torch.nn.Linear(in_features=num_features, out_features=1, bias=True)
        if fc_path:
            if os.path.exists(fc_path):
                self.fc.load_state_dict(torch.load(fc_path))

    def forward(self, features):
        return self.fc(features)

    def save(self, save_dir):
        torch.save(self.fc.state_dict(), os.path.join(save_dir, "fc_weight.bin"))
