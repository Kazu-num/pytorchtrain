import torch
from torch import nn

a = torch.randn(1, 16, 24, 128)
# print(a)

# print(a.shape)
# print(a[0][0][0][4])
# out = a[:, :, :, 0:9223372036854775807:2]
# print(out.shape)
# print(out[0][0][0][2])

class Slice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :, :, ::2]
    
model = Slice()
# out = model(a)
# print(out.shape)
# print(out[0][0][0][2])

torch.onnx.export(
    model,
    a,
    'slice_test.onnx',
    input_names=['input'],
    output_names=['output']
)