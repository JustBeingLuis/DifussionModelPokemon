import os
import torch
import torchvision
from unet import UNet
from diffusion import DiffusionProcess

torch.manual_seed(42)

def generate(n_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generando con {device}")

    unet = UNet(in_channels=3,out_channels=3).to(device)
    unet.load_state_dict(torch.load("pokemon_unet.pth", weights_only=True))

    difussion = DiffusionProcess(timesteps=100, device=device)

    n_samples = n_samples
    print(f"Generando {n_samples} pokemones desde ruido puro...")
    samples = difussion.sample(unet, n_samples)

    os.makedirs("output", exist_ok=True)

    file_name = "output/pokemones_generados_02.png"
    torchvision.utils.save_image(samples, file_name, nrow=4)
    print(f"Generacion terminada, guardado en {file_name}")

if __name__ == "__main__":
    generate(16)
