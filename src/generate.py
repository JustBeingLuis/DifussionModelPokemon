import os
import torch
import torchvision
from unet import UNet
from diffusion import DiffusionProcess
import argparse
import math

#torch.manual_seed(12)

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generando con {device}")

    unet = UNet(in_channels=3,out_channels=3).to(device)
    unet.load_state_dict(torch.load("pokemon_unet.pth", weights_only=True))

    difussion = DiffusionProcess(timesteps=args.steps, device=device)

    
    print(f"Generando {args.samples} pokemones desde ruido puro...")
    samples = difussion.sample(unet, args.samples)

    os.makedirs("output", exist_ok=True)

    grid_rows = math.ceil(math.sqrt(args.samples))
    torchvision.utils.save_image(samples, args.output, nrow=grid_rows)
    print(f"Generacion terminada, guardado en {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para generar Pokemones")

    parser.add_argument("--samples", type=int, default=16, help="Cantidad de imagenes a generar")
    parser.add_argument("--steps", type=int, default=100, help="Pasos para resolver la ODE")
    parser.add_argument("--output", type=str, default="output/generados.png", help="Nombre del archivo final")

    args = parser.parse_args()

    generate(args)