import torch
import torch.nn.functional as F
from torch import nn
import hiddenlayer as hl

def main():
    model = nn.Sequential(
        nn.Conv2d(1, 20, 5),
        nn.ReLU(),
        nn.Conv2d(20, 64, 5),
        nn.ReLU()
    )
    x = torch.randn([1, 1, 28, 28])
    y = model(x)
    g = hl.build_graph(model, torch.zeros([1, 1, 28, 28]))
    g.save("C:\example.png", format="png")

if __name__ == '__main__':
    main()