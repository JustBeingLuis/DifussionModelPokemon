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

    diffusion = DiffusionProcess(timesteps=50, device=device)

    unet = UNet(in_channels=3,out_channels=3).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 2000

    for epoch in tqdm(range(epochs), desc="Entrenando modelo..."):
        for step, batch in enumerate(dataloader):

            x_0 = batch.to(device)
            u = torch.randn((x_0.shape[0],), device=device)
            t = torch.sigmoid(u)

            z_t, true_v = diffusion.add_noise(x_0,t)

            t_tensor = t * 1000.0

            predicted_x0 = unet(z_t, t_tensor)

            t_exp = t.view(-1, 1, 1, 1)
            predicted_v = (predicted_x0 - z_t) / (1.0 - t_exp + 1e-5)

            loss = criterion(predicted_v, true_v)

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
