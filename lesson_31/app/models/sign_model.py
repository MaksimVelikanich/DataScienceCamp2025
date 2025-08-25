import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=784, output_size=24):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
