import torch 
from torch import nn
from torch.nn import functional as F
import math
import os
import shutil
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms


class SelfAttention(nn.Module):
  def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
    super().__init__()
    self.n_heads = n_heads
    self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
    self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)

    self.d_heads = embd_dim // n_heads

  def forward(self, x, casual_mask=False):

    batch_size, seq_len, d_emed = x.shape

    interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

    q, k, v = self.in_proj(x).chunk(3, dim=-1)


    q = q.view(interim_shape)
    k = k.view(interim_shape)
    v = v.view(interim_shape)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    weight = q @ k.transpose(-1, -2)

    if casual_mask:
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight.masked_fill_(mask, -torch.inf)

    weight /= math.sqrt(self.d_heads)

    weight = F.softmax(weight, dim=-1)

    output = weight @ v
    output = output.transpose(1, 2)
    output = output.reshape((batch_size, seq_len, d_emed))

    output = self.out_proj(output)

    return output

class AttentionBlock(nn.Module):
  def __init__(self, channels):
      super().__init__()
      self.groupnorm = nn.GroupNorm(32, channels)
      self.attention = SelfAttention(1, channels)

  def forward(self, x):

      residual = x.clone()

      x = self.groupnorm(x)
      n, c, h, w = x.shape
      x = x.view((n, c, h * w))

      x = x.transpose(-1, -2)

      x = self.attention(x)
      x = x.transpose(-1, -2)
      x = x.view((n, c, h, w))
      x += residual

      return x

"""This Implements ResidualBlock which we use in Encoder and Decoder"""

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.groupnorm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

  def forward(self, x):
    residue = x.clone()

    x = self.groupnorm1(x)
    x = F.selu(x)
    x = self.conv1(x)
    x = self.groupnorm2(x)
    x = self.conv2(x)

    return x + self.residual_layer(residue)

class Encoder(nn.Sequential):
    def  __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x):

        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        x = mean + eps * std
        x *= 0.18215
        return x

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x



# def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
#     image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)

#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     for file in train_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

#     for file in test_files:
#         shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

#     print(f"Dataset split complete. {len(train_files)} training images, {len(test_files)} test images.")

# source_dir = "./all-dogs"
# train_dir = "./data/train/dogs"
# test_dir = "./data/test/dogs"

# #split_dataset(source_dir, train_dir, test_dir)

# """# Implement VAE"""

# class VAE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded, encoded

# """# Train The VAE"""



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# from torchvision import transforms



# device = torch.device('cuda')


# num_epochs = 100
# learning_rate = 1e-4
# beta = 0.00025


# transform = transforms.Compose([
#     transforms.Resize((56, 56)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# batch_size = 10
# dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# accumulation_steps = 1
# effective_batch_size = batch_size * accumulation_steps

# train_losses = []


# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for i, (images, _) in enumerate(dataloader):
#         images = images.to(device)  
#        # print(f"Input images are on: {images.device}")  

#         reconstructed, encoded = model(images)
#        # print(f"Reconstructed output is on: {reconstructed.device}") 
#         torch.cuda.synchronize()

#         recon_loss = nn.MSELoss()(reconstructed, images)
#         torch.cuda.synchronize()

#         mean, log_variance = torch.chunk(encoded, 2, dim=1)
#         #print(f"Mean is on: {mean.device}, Log variance is on: {log_variance.device}")  
#         kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
#         loss = recon_loss + beta * kl_div

#         loss = loss / accumulation_steps
#         loss.backward()
#         torch.cuda.synchronize()

#         if (i + 1) % accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         train_loss += loss.item() * accumulation_steps

#         print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
#               f'Loss: {loss.item()*accumulation_steps:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')

#         with torch.no_grad():

#             sample_image = images[0].unsqueeze(0)
#             sample_reconstructed = model(sample_image)[0]

#             sample_image = (sample_image * 0.5) + 0.5
#             sample_reconstructed = (sample_reconstructed * 0.5) + 0.5

#             if (epoch + 1) % 5 == 0:  # Save every 5 epochs
#               torchvision.utils.save_image(sample_reconstructed, f'reconstructed_epoch_{epoch+1}.png')

#     train_losses.append(train_loss / len(dataloader))
#     val_loss = loss.item()
#     best_val_loss = float('inf')
#     if val_loss < best_val_loss:
#       best_val_loss = val_loss
#       torch.save(model.state_dict(), f'vae_best_model_num{epoch+1}.pth')
#     torch.save(model.state_dict(), f'vae_model_epoch_{epoch+1}.pth')

# print('Training finished!')


"""# Combining with Stable Diffusion

"""

class CustomVAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def encode(self, x):
    return self.encoder(x)

  def decode(self,z):
    return self.decoder(z)

  def forward(self, x):
    z = self.encode(x)
    x_reconstructed = self.decode(z)
    return x_reconstructed

  def load_pretrained_weights(self, weight_path):
    self.load_state_dict(torch.load(weight_path))

from diffusers import AutoencoderKL

class DiffusersCompatibleVAE(AutoencoderKL):
  def __init__(self, vae):
    super().__init__()
    self.vae = vae

  def encode(self, x):
    mean, log_var = self.vae.encoder(x)
    print("mean")
    return mean, log_var

  def decode(self,  z, **kwargs):
    print("Input shape:", z.shape)
    out = self.vae.decoder(z).unsqueeze(0)
    print(out.shape)
    return out


AutoencoderKL.from_pretrained("./model/vae")

import torch

weights = torch.load("./vae_model_epoch_60.pth", map_location="cpu")

new_weights = {}

for k,v in weights.items():

  new_key = k.replace("_","")

  new_key = new_key.replace("residuallayer", "residual_layer") \
                   .replace("inproj", "in_proj") \
                   .replace("outproj", "out_proj")

  new_weights[new_key] = v

vae = CustomVAE()
# vae.load_state_dict(new_weights)

device = torch.device("cuda")
vae = vae.to(device)

from diffusers import StableDiffusionPipeline
import torch

vae = CustomVAE()
compatible_vae = DiffusersCompatibleVAE(vae=vae)
pipe = StableDiffusionPipeline.from_pretrained("./model")
pipe = pipe.to("cuda")

#Finallly

prompt= "a pomerenian dog that is happy"
output_dir = "./output"
image = pipe(prompt, num_inference_steps=50).images[0]
output_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_steps_20.png")
image.save(output_path)

print(f"Image saved to {output_path}")

