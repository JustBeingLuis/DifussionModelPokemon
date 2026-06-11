import torch
from tqdm import tqdm
import torch.nn as nn
from dataset import get_dataloader
from diffusion import DiffusionProcess
from unet import UNet

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando con {device}")

    dataloader = get_dataloader()

    diffusion = DiffusionProcess(timesteps=1000, device=device)

    unet = UNet(in_channels=3,out_channels=3).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 2000

    for epoch in tqdm(range(epochs), desc= "Entrenando modelo...", total= self.timesteps):
        for step, batch in enumerate(dataloader):

            x_0 = batch.to(device)
            actual_batch = x_0.shape[0]

            t = torch.randint(0, diffusion.timesteps, (actual_batch,), device=device).long()

            x_t, true_noise = diffusion.add_noise(x_0,t)

            predicted_noise = unet(x_t,t)

            loss = criterion(predicted_noise,true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Época [{epoch}/{epochs}] - Error (Loss): {loss.item():.4f}")

    print("Guardando el modelo ....")
    torch.save(unet.state_dict(), "pokemon_unet.pth")
    print("Modelo guardado correctamente")
    
if __name__ == "__main__":
    train()
