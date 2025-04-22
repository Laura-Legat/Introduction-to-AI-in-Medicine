import torch
import imageio


class Watermark(torch.nn.Module):
    def __init__(self, path: str, shape: int = 64, label=0):
        super().__init__()
        watermark = imageio.imread(path)
        self.template = torch.zeros((shape, shape))
        h, w = watermark.shape
        self.template[..., -2-h:-2, -2-w:-2] = 1-torch.as_tensor(watermark, dtype=torch.float32)/255
        self.label = label

    def forward(self, inputs):
        x, y = inputs
        x[y[:, 0] == self.label, ...] = (1-self.template) * x[y[:, 0] == self.label, ...] + self.template * -.75
        return x, y
