
import torch
import torch.nn as nn

class Regression(nn.Module):

    def __init__(self, input_feature, num_classes=1):
        super(Regression, self).__init__()
        self.classifier = nn.Linear(in_features=input_feature, out_features=num_classes)

    def forward(self, x):
        out = self.classifier(x)

        return out