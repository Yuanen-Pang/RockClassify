import torch
import torch.nn as nn

class LinearClassifier(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, hlayer_num) -> None:
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(in_dim, h_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(h_dim, h_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(h_dim, out_dim)
        # )
        layers = []
        for idx, num in enumerate(range(hlayer_num)):
            if idx == 0: 
                layers.append(nn.Linear(in_dim, h_dim))
            else:
                layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
        if len(layers) == 0: h_dim = in_dim
        layers.append(nn.Linear(h_dim, out_dim))
        self.model = nn.Sequential(*layers)
        # self.model = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    inp = torch.rand((8, 512))
    model = LinearClassifier(512, 256, 3)
    print(model(inp).shape)
