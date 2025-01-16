import torch as pt
from torch import nn, Tensor
from torch.nn import functional as F
from math import log

class ResidualBlock(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.act1 = nn.GELU(approximate='tanh')
        self.norm2 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act2 = nn.GELU(approximate='tanh')

    def forward(self, x:Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x + residual)
        return x


class SelfAttention(nn.Module):

    def __init__(self, channels:int, heads=4, resolution=16):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)

    def forward(self, x:Tensor) -> Tensor:
        x = x.view(-1,self.channels, self.resolution * self.resolution).swapaxes(1, 2) #(B,H*W,C)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        attention_value = attention_value + x #(B,H*W,C)

        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.resolution, self.resolution)


class conditional_Unet(nn.Module):

    def __init__(self, channels:int, heads:int, num_classes=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.num_classes = num_classes
        # Time
        self.time_emb_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
        )
        # Class condition
        self.class_emb_layer = nn.Embedding(num_classes, channels)

        # Downsampling
        self.first_conv = nn.Conv2d(3, channels, kernel_size=1)
        self.res_L1 = ResidualBlock(channels)
        self.down_L1 = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.res_L2 = ResidualBlock(channels*2)
        self.att_L2 = SelfAttention(channels*2, heads=heads, resolution=16)
        self.down_L2 = nn.Conv2d(channels*2, channels*2, kernel_size=2, stride=2)

        # Middle
        self.res_M1 = ResidualBlock(channels*3)
        self.att_M1 = SelfAttention(channels*3, heads=heads, resolution=8)
        self.res_M2 = ResidualBlock(channels*3)
        self.att_M2 = SelfAttention(channels*3, heads=heads, resolution=8)
        self.res_M3 = ResidualBlock(channels*3)
        self.att_M3 = SelfAttention(channels*3, heads=heads, resolution=8)

        # Upsampling
        self.up_R2 = nn.ConvTranspose2d(channels*3, channels, kernel_size=2, stride=2)
        self.res_R2 = ResidualBlock(channels*2)
        self.att_R2 = SelfAttention(channels*2, heads=heads, resolution=16)
        self.up_R1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.res_R1 = ResidualBlock(channels*2)
        self.last_conv = nn.Conv2d(channels*2, 3, kernel_size=1)

    def get_sinusoidal_embedding(self, t:Tensor, emb_dim:int) -> Tensor:
        """
        t: (B,1)
        """
        half_dim =  emb_dim // 2
        frequencies = pt.exp(-log(10000.0) * pt.arange(half_dim, dtype=pt.float32) / half_dim).to(t.device)
        sinusoidal = pt.cat([pt.sin(t * frequencies), pt.cos(t * frequencies)], dim=1)
        return sinusoidal

    def forward(self, x:Tensor, t:Tensor, labels:Tensor) -> Tensor:
        '''x: (B,3,32,32), t:(B,1)'''

        time_emb = self.get_sinusoidal_embedding(t, self.channels) # (B,C)
        time_emb = self.time_emb_layer(time_emb)[..., None, None] # (B,C,1,1)
        if labels is not None:
            label_emb = self.class_emb_layer(labels)[..., None, None] # (B,C,1,1)
            time_emb += label_emb # (B,C,1,1)

        # Downsampling
        x = self.first_conv(x) # (B,C,32,32)
        x_L1 = self.res_L1(x) # (B,C,32,32)
        x_L2 = self.down_L1(x_L1) # (B,C,16,16)
        x_L2 = pt.cat((x_L2, time_emb.repeat(1,1,16,16)), dim=1) # (B,C+C,16,16)
        # import code; code.interact(local=locals())
        x_L2 = self.att_L2(self.res_L2(x_L2)) # (B,2C,16,16)
        x_L3 = self.down_L2(x_L2) # (B,2C,8,8)
        x_L3 = pt.cat((x_L3, time_emb.repeat(1,1,8,8)), dim=1) # (B,2C+C,8,8)
        
        #Middle
        x_M1 = self.att_M1(self.res_M1(x_L3)) # (B,3C,8,8)
        x_M2 = self.att_M2(self.res_M2(x_M1)) # (B,3C,8,8)
        x_M3 = self.att_M3(self.res_M3(x_M2)) # (B,3C,8,8)

        # Upsampling
        x_R2 = self.up_R2(x_M3) # (B,C,16,16)
        x_R2 = pt.cat((x_R2, time_emb.repeat(1,1,16,16)), dim=1) # (B,C+C,16,16)
        x_R2 = x_R2 + x_L2 # test (B,C+C,16,16)
        x_R2 = self.att_R2(self.res_R2(x_R2)) # (B,2C,16,16)
        x_R1 = self.up_R1(x_R2) # (B,C,32,32)
        x_R1 = x_R1 + x_L1 # (B,C,32,32)
        x_R1 = pt.cat((x_R1, time_emb.repeat(1,1,32,32)), dim=1) # (B,C+C,32,32)
        x_R1 = self.res_R1(x_R1) # (B,2C,32,32)
        y = self.last_conv(x_R1) # (B,3,32,32)

        return y
    

class Unet(nn.Module):

    def __init__(self, channels:int, heads:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        # Time
        self.time_emb_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
        )

        # Downsampling
        self.first_conv = nn.Conv2d(3, channels, kernel_size=1)
        self.res_L1 = ResidualBlock(channels)
        self.down_L1 = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.res_L2 = ResidualBlock(channels*2)
        self.att_L2 = SelfAttention(channels*2, heads=heads, resolution=16)
        self.down_L2 = nn.Conv2d(channels*2, channels*2, kernel_size=2, stride=2)

        # Middle
        self.res_M1 = ResidualBlock(channels*3)
        self.att_M1 = SelfAttention(channels*3, heads=heads, resolution=8)
        self.res_M2 = ResidualBlock(channels*3)
        self.att_M2 = SelfAttention(channels*3, heads=heads, resolution=8)
        self.res_M3 = ResidualBlock(channels*3)
        self.att_M3 = SelfAttention(channels*3, heads=heads, resolution=8)

        # Upsampling
        self.up_R2 = nn.ConvTranspose2d(channels*3, channels, kernel_size=2, stride=2)
        self.res_R2 = ResidualBlock(channels*2)
        self.att_R2 = SelfAttention(channels*2, heads=heads, resolution=16)
        self.up_R1 = nn.ConvTranspose2d(channels*2, channels, kernel_size=2, stride=2)
        self.res_R1 = ResidualBlock(channels*2)
        self.last_conv = nn.Conv2d(channels*2, 3, kernel_size=1)

    def get_sinusoidal_embedding(self, t:Tensor, emb_dim:int) -> Tensor:
        """
        t: (B,1)
        """
        half_dim =  emb_dim // 2
        frequencies = pt.exp(-log(10000.0) * pt.arange(half_dim, dtype=pt.float32) / half_dim).to(t.device)
        sinusoidal = pt.cat([pt.sin(t * frequencies), pt.cos(t * frequencies)], dim=1)
        return sinusoidal

    def forward(self, x:Tensor, t:Tensor, *args, **kwargs) -> Tensor:
        '''x: (B,3,32,32), t:(B,1)'''

        time_emb = self.get_sinusoidal_embedding(t, self.channels) # (B,C)
        time_emb = self.time_emb_layer(time_emb)[..., None, None] # (B,C,1,1)

        # Downsampling
        x = self.first_conv(x) # (B,C,32,32)
        x_L1 = self.res_L1(x) # (B,C,32,32)
        x_L2 = self.down_L1(x_L1) # (B,C,16,16)
        x_L2 = pt.cat((x_L2, time_emb.repeat(1,1,16,16)), dim=1) # (B,C+C,16,16)
        # import code; code.interact(local=locals())
        x_L2 = self.att_L2(self.res_L2(x_L2)) # (B,2C,16,16)
        x_L3 = self.down_L2(x_L2) # (B,2C,8,8)
        x_L3 = pt.cat((x_L3, time_emb.repeat(1,1,8,8)), dim=1) # (B,2C+C,8,8)
        
        #Middle
        x_M1 = self.att_M1(self.res_M1(x_L3)) # (B,3C,8,8)
        x_M2 = self.att_M2(self.res_M2(x_M1)) # (B,3C,8,8)
        x_M3 = self.att_M3(self.res_M3(x_M2)) # (B,3C,8,8)

        # Upsampling
        x_R2 = self.up_R2(x_M3) # (B,C,16,16)
        x_R2 = pt.cat((x_R2, time_emb.repeat(1,1,16,16)), dim=1) # (B,C+C,16,16)
        x_R2 = x_R2 + x_L2 # test (B,C+C,16,16)
        x_R2 = self.att_R2(self.res_R2(x_R2)) # (B,2C,16,16)
        x_R1 = self.up_R1(x_R2) # (B,C,32,32)
        x_R1 = x_R1 + x_L1 # (B,C,32,32)
        x_R1 = pt.cat((x_R1, time_emb.repeat(1,1,32,32)), dim=1) # (B,C+C,32,32)
        x_R1 = self.res_R1(x_R1) # (B,2C,32,32)
        y = self.last_conv(x_R1) # (B,3,32,32)

        return y


if __name__ == "__main__":
    device = pt.device("cuda")
    model = Unet(channels=256, heads=4).to(device)

    x = pt.randn(size=(4,3,32,32)).to(device)
    t = pt.rand(size=(4,1), device=device)
    labels = pt.randint(0, 10, size=(4,), device=device)
    out = model(x, t=t, labels=labels)
    print(out.shape)

    import code; code.interact(local=locals())