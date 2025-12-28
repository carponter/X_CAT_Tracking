import torch
import torch.nn as nn
import numpy as np

class FlattenMlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=nn.ReLU(),
            output_activation=None,
    ):
        super(FlattenMlp, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.layers = []
        in_size = input_size
        
        for next_size in hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.layers.append(fc)
            self.layers.append(hidden_activation)
            
        fc = nn.Linear(in_size, output_size)
        self.layers.append(fc)
        if output_activation is not None:
            self.layers.append(output_activation)
            
        self.model = nn.Sequential(*self.layers)

    def forward(self, *inputs):
        flat_inputs = torch.cat([x.view(x.size(0), -1) for x in inputs], dim=1)
        return self.model(flat_inputs)