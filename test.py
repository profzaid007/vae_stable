import torch
from torch import nn
from torch.nn import functional as F
import math
import os
import shutil
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import transforms
from diffusers import AutoencoderKL, StableDiffusionPipeline
# from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt

# Constants
DEVICE = torch.device('cuda')
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
BETA = 0.00025
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
IMAGE_SIZE = (56, 56)  # Adjust as needed
SOURCE_DIR = "./all-dogs"
TRAIN_DIR = "./data/train/dogs"
TEST_DIR = "./data/test/dogs"
MODEL_SAVE_PATH = "./vae_model.pth"


# Model Definitions
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

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).reshape((batch_size, seq_len, d_emed))
        return self.out_proj(output)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        residual = x.clone()
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w)).transpose(-1, -2)
        x = self.attention(x).transpose(-1, -2).view((n, c, h, w))
        return x + residual


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residue = x
        x = self.groupnorm1(x)
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        return x + self.residual_layer(residue)


class Encoder(nn.Sequential):
    def __init__(self):
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
        return x * 0.18215


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


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class CustomVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class DiffusersCompatibleVAE(AutoencoderKL):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def encode(self, x):
        mean, log_var = self.vae.encoder(x)
        return mean, log_var

    def decode(self, z, **kwargs):
        return self.vae.decoder(z)


# Dataset Preparation
def split_dataset(source_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))
    print(f"Dataset split complete. {len(train_files)} training images, {len(test_files)} test images.")


def load_dataset(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Training Functions
def train_vae(model, dataloader, num_epochs, learning_rate, beta, accumulation_steps, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            reconstructed, encoded = model(images)
            recon_loss = nn.MSELoss()(reconstructed, images)
            mean, log_variance = torch.chunk(encoded, 2, dim=1)
            kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
            loss = (recon_loss + beta * kl_div) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'Loss: {loss.item()*accumulation_steps:.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_div.item():.4f}')

        train_losses.append(train_loss / len(dataloader))
        torch.save(model.state_dict(), f'vae_model_epoch_{epoch+1}.pth')

    return train_losses


def plot_losses(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Loss over Time')
    plt.legend()
    plt.show()


# Integration with Stable Diffusion
def load_pretrained_vae(model, weight_path):
    weights = torch.load(weight_path, map_location="cpu")
    new_weights = {}

    for k, v in weights.items():
        new_key = k.replace("_", "").replace("residuallayer", "residual_layer") \
                     .replace("inproj", "in_proj").replace("outproj", "out_proj")
        new_weights[new_key] = v

    model.load_state_dict(new_weights)
    return model


def generate_image_with_stable_diffusion(prompt, vae, device):
    compatible_vae = DiffusersCompatibleVAE(vae=vae)
    pipe = StableDiffusionPipeline.from_pretrained("./model/StableDiffusion-v1.5.diff", vae=compatible_vae)
    pipe = pipe.to(device)
    image = pipe(prompt, num_inference_steps=60).images[0]
    return image


# Main Function
def main():
    # Prepare dataset
    #split_dataset(SOURCE_DIR, TRAIN_DIR, TEST_DIR)
    train_dataloader = load_dataset(TRAIN_DIR, IMAGE_SIZE, BATCH_SIZE)

    # Initialize and train VAE
    model = VAE().to(DEVICE)
    train_losses = train_vae(model, train_dataloader, NUM_EPOCHS, LEARNING_RATE, BETA, ACCUMULATION_STEPS, DEVICE)
    plot_losses(train_losses)

    # Save the final model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Load pretrained VAE and generate an image
    vae = CustomVAE().to(DEVICE)
    vae = load_pretrained_vae(vae, MODEL_SAVE_PATH)
    prompt = "human"
    image = generate_image_with_stable_diffusion(prompt, vae, DEVICE)
    image.save("generated_image.png")


if __name__ == "__main__":
    main()