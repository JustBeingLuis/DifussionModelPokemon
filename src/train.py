import os
import torch
from tqdm import tqdm
import torch.nn as nn
from dataset import get_dataloader
from diffusion import DiffusionProcess
from unet import UNet
import argparse
import config
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando con {device}")

    dataloader = get_dataloader(args.batch_size)

    diffusion = DiffusionProcess(timesteps=config.TIMESTEPS, device=device)

    unet = UNet(in_channels=3,out_channels=3).to(device)

    epochs = args.epochs

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)
    #schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    

    start_epoch = 0
    os.makedirs("checkpoints", exist_ok=True)

    if args.resume:
        print(f"Reanudando desde: {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=True)
        unet.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        #schedular.load_state_dict(checkpoint["schedular_state"])
        start_epoch = checkpoint["epoch"] + 1

    writer = SummaryWriter("runs/pokemon_experiment")
    for epoch in tqdm(range(start_epoch,epochs), desc="Entrenando modelo..."):
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
            
        #schedular.step()

        if epoch % 100 == 0:
            print(f"Época [{epoch}/{epochs}] - Error (Loss): {loss.item():.4f}")
            writer.add_scalar("Loss/train", loss.item(),epoch)
            #writer.add_scalar("Learning_Rate", schedular.get_last_lr()[0],epoch)

            checkpoint = {
                'epoch': epoch,
                'model_state': unet.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                #'schedular_state': schedular.state_dict()
            }

            torch.save(checkpoint, f"checkpoints/epoch_{epoch}.pt")
            print(f"Checkpoint guardado: checkpoints/epoch_{epoch}.pt")

    

    print("Guardando el modelo ....")
    torch.save(unet.state_dict(), "pokemon_unet.pth")
    writer.close()
    print("Modelo guardado correctamente")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Entrenamiento")
    
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Épocas totales")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Tamaño del paquete")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning Rate del Optimizador")

    #Checkpoints
    parser.add_argument("--resume", type=str, default="", help="Ruta al checkpoint para reanudar (.pt)")

    args = parser.parse_args()
    
    train(args)

