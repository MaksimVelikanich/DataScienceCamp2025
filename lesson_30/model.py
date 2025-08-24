import torch
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
        logits = self.linear_relu_stack(x)
        return logits


def load_model(weights_path: str, device: torch.device):
    model = SignLanguageModel(input_size=784, output_size=24)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
