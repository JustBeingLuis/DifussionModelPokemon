import torch
from tqdm import tqdm

class DiffusionProcess:
    def __init__(self, timesteps=100, device="cpu"):
        self.timesteps=timesteps
        self.device=device
    
    def add_noise(self,x_0,t):
        noise = torch.randn_like(x_0).to(self.device)

        t = t.view(-1,1,1,1)
        z_t = t * x_0 + (1.0 - t) * noise
        v = x_0 - noise

        return z_t, v 

    @torch.no_grad()
    def sample(self, unet, n_samples=1, img_size = 64, channels = 3 ):

        unet.eval()
        
        z = torch.randn((n_samples, channels,img_size,img_size)).to(self.device)

        dt = 1.0 / self.timesteps


        for step in tqdm(range(self.timesteps), desc="Generando..."):
            t_current = step * dt
            t_next = (step + 1) * dt


            t_tensor_current = torch.full((n_samples,), t_current * 1000, device=self.device, dtype=torch.float32)
            t_tensor_next = torch.full((n_samples,), t_next * 1000, device=self.device, dtype=torch.float32)

            # 1. Medimos la velocidad en la posición actual
            # Como la red fue entrenada para escupir x_0, la convertimos a velocidad
            x0_pred_1 = unet(z, t_tensor_current)
            v1 = (x0_pred_1 - z) / (1.0 - t_current + 1e-5)

            if step == self.timesteps -1:
                z = z + (v1*dt)
                break

            z_tmp = z + (v1 * dt)
            
            # 3. Lo mismo para el viaje al futuro
            x0_pred_2 = unet(z_tmp, t_tensor_next)
            v2 = (x0_pred_2 - z_tmp) / (1.0 - t_next + 1e-5)

            v_mean = (v1 + v2) / 2.0
            z = z + (v_mean * dt)

        z = (z.clamp(-1,1) + 1) / 2
        return z
        

if __name__ == "__main__":

    from dataset import get_dataloader

    diffusion = DiffusionProcess(device="cpu")
    loader = get_dataloader()
    x_0 = next(iter(loader))

    batch_size = x_0.shape[0]
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device="cpu")

    x_t, noise = diffusion.add_noise(x_0,t)

    print(f"Tensor Original Limpio (x_0)  : {x_0.shape}")
    print(f"Vector de Timesteps (t)       : {t.shape} (Valores de 0 a 999)")
    print(f"Tensor de Ruido Puro (noise)  : {noise.shape}")
    print(f"Tensor Destruido Final (x_t)  : {x_t.shape}")
    print(f"\nVarianza Original : Max {x_0.max():.4f}, Min {x_0.min():.4f}")
    print(f"Varianza Destruida: Max {x_t.max():.4f}, Min {x_t.min():.4f}")