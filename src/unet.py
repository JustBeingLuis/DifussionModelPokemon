import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim

    def forward(self,time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device = device)*-embeddings)

        embeddings = time[:,None] * embeddings[None,:]

        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)

        self.norm = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.block1 = Block(in_channels, out_channels)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim,out_channels)
        )

        self.block2 = Block(out_channels, out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self,x,time_emb):

        h = self.block1(x)

        t = self.time_mlp(time_emb)
        t = t[: ,: , None, None]

        h = h + t

        h = self.block2(h)

        return h + self.residual_conv(x)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1)

        self.proj_out = nn.Conv2d(channels,channels, kernel_size=1)

    def forward(self, x):

        B,C,H,W = x.shape

        qkv = self.qkv(x).view(B,3,C,H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] 

        attn = (q.transpose(-2,-1) @ k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2,-1))

        out = out.view(B,C,H,W)

        return x + self.proj_out(out)

class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels=3, time_dim = 256):
        super().__init__()

        # Capa de entrada
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3,padding=1)

        # Inicializamos el embebido de los tiempos
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64,time_dim),
            nn.GELU(),
            nn.Linear(time_dim,time_dim)
        )

        # Encoder
        self.downs = nn.ModuleList([
            ResnetBlock(64,128,time_dim),
            ResnetBlock(128,256,time_dim)
        ])

        self.down_pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid_block1 = ResnetBlock(256,256,time_dim)

        # Capas de atencion, ya que hemos redujimos la dimensionalidad pero a nivel de canales (caracteristicas hay bastantes)
        self.mid_attn = Attention(256)
        self.mid_block2 = ResnetBlock(256,256,time_dim)

        # Decoder
        self.ups = nn.ModuleList([
            ResnetBlock(256 + 256, 128, time_dim),
            ResnetBlock(128 + 128, 64, time_dim)
        ])

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self,x,time):
        t = self.time_mlp(time)

        x = self.init_conv(x)

        skip1 = self.downs[0](x,t)

        x = self.down_pool(skip1)

        skip2 = self.downs[1](x,t)

        x = self.down_pool(skip2)

        # BottleNeck
        x = self.mid_block1(x,t)
        x = self.mid_attn(x)
        x = self.mid_block2(x,t)

        # Decoder

        x = self.up_sample(x)
        x = torch.cat((x,skip2),dim=1)

        x = self.ups[0](x,t)


        x = self.up_sample(x)
        x = torch.cat((x,skip1),dim=1)   

        x = self.ups[1](x, t) 

        return self.final_conv(x)        