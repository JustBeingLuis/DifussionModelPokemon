import torch
from tqdm import tqdm

class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps=timesteps
        self.device=device

        # Cantidad de ruido a agregar en cada paso
        self.betas = torch.linspace(beta_start,beta_end,timesteps).to(device)

        # Cantidad de informacion que se conserva de la imagen original
        self.alphas = 1.0 - self.betas

        # Cantidad de foto que sobrevive en cada uno de los pasos
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        # Usamos la raiz cuadrada para evitar el problema de los cuadrados al mezclar variables
        # Raiz de lo que sobrevive de la foto en cada paso
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)

        # Raiz de todo el ruido acumulado desde el paso 0 hasta t
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
    
    def add_noise(self,x_0,t):
        noise = torch.randn_like(x_0).to(self.device)

        sqrt_alpha_bar = self.sqrt_alpha_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].view(-1,1,1,1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_cumprod * noise  

        return x_t, noise 

    @torch.no_grad()
    def sample(self, unet, n_samples=1, img_size = 64, channels = 3 ):

        unet.eval()
        
        x = torch.randn((n_samples, channels,img_size,img_size)).to(self.device)

        for i in tqdm(reversed(range(0, self.timesteps)), desc = "Generando..."):
            t = torch.full((n_samples,),i,device=self.device, dtype=torch.long)    
            pred_noise = unet(x,t)    

            alpha_t = self.alphas[t].view(-1,1,1,1)
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1,1,1,1)

            x = (1.0 / torch.sqrt(alpha_t)) * (x - ((1- alpha_t) / sqrt_one_minus_alpha_cumprod_t)* pred_noise)

            # Le agragamos ruido leve en cada iteracion para que la imagen no sea suave o sin textura
            if i > 0:
                noise = torch.randn_like(x)
                beta_t = self.betas[t].view(-1,1,1,1)
                x = x + torch.sqrt(beta_t)* noise
        
        x = (x.clamp(-1,1) + 1) / 2
        return x
        

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