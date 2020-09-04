import torch
import os


class LinearModel(torch.nn.Module):
    def __init__(self, in_features, fc_path=None):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1, bias=False)
        if fc_path:
            self.linear.load_state_dict(torch.load(fc_path))

    def forward(self, input_feature):
        """
        forward
        :param input_feature: batch_size * num_features
        :return:
        """
        scores = self.linear(input_feature)
        return scores  # batch_size * 1

    def save(self, save_dir):
        torch.save(self.linear.state_dict(), os.path.join(save_dir, "fc_weight.bin"))
