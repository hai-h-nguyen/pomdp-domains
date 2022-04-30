import torch
import torch.nn as nn


class MinigridFeatureExtractor(nn.Module):
    """one-layer MLP with relu
    Used for extracting features for Minigrid-like observations, based on
    https://github.com/lcswillems/rl-starter-files/blob/master/model.py
    """
    def __init__(self, n, m):
        super(MinigridFeatureExtractor, self).__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self._output_size = ((n-1)//2-2)*((m-1)//2-2)*64

    def forward(self, inputs):
        length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = inputs.reshape(-1, 7, 7, 3)        
        x = inputs.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        return x.reshape(length, batch_size, self._output_size)

# input = torch.rand((10, 7, 7, 3))
input = torch.rand((101, 32, 7, 7, 3)).to("cuda:0")  # 101, 32, 3, 7, 7
encoder = MinigridFeatureExtractor(7, 7).to("cuda:0")

output = encoder(input)

print(output.shape, encoder._output_size)