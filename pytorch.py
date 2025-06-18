# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.version.cuda)  # Should return the CUDA version

# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.backends.cudnn.enabled)
import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

# Create model and move to GPU
model = SimpleModel().to(device)
print(f"Model is on: {next(model.parameters()).device}")

# Create input tensor and move to GPU
x = torch.randn(5, 10).to(device)
print(f"Input tensor is on: {x.device}")

# Forward pass
output = model(x)
print(f"Output tensor is on: {output.device}")