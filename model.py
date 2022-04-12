import torch
import torch.nn as nn
import torch.nn.functional as F

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, output_ch=1, skips=[4]):
        """ 
        default: D=8, W=256, input_ch=63, output_ch=1, skips=[4]
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):

        input_pts = x.float()
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = F.relu(self.output_linear(h))

        return outputs   