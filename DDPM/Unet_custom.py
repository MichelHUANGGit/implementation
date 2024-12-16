
import torch as pt
from torch import nn, Tensor
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels:int, size:int, heads:int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B, C, H*W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(B, C, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, emb_dim=256):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1) # -8 pixels
        # output_size = (input_size + 2*pad - kernel_size) / stride + 1 = (input_size - 9) + 1
        self.conv = nn.Sequential(
            DoubleConv(out_channels, out_channels, residual=True),
            DoubleConv(out_channels, out_channels, residual=True),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.down(x)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, emb_dim=256):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1)
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(out_channels+skip_channels, out_channels, residual=False),
            DoubleConv(out_channels, out_channels, residual=True),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = pt.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet32x32_custom(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        img_size = 32
        # C = (c_in, 5, 10, 32, 48, 64, 96, 128, 196, 256)
        # multiplier = 1.5
        # old_C = 16; new_C = round(old_C * multiplier)
        kernel_size = 9

        self.conv_in = nn.Conv2d(3, 32, kernel_size=1)

        # Down_block 1
        self.sa11 = SelfAttention(32, 32, heads=1) # -> (B,3,32,32)
        self.sa12 = SelfAttention(32, 32, heads=1) # -> (B,3,32,32)
        self.down1 = Down(32, 48, kernel_size) # -> (B,5,24,24)

        # Down_block 2
        # self.conv2 = DoubleConv(5, 5, residual=True) # -> (B,5,24,24)
        self.sa21 = SelfAttention(48, 24, heads=1) # -> (B,5,24,24)
        self.sa22 = SelfAttention(48, 24, heads=1) # -> (B,5,24,24)
        self.down2 = Down(48, 96, kernel_size) # -> (B,10,16,16)

        # Down block 3
        # self.conv3 = DoubleConv(10, 10, residual=True) # -> (B,10,16,16)
        self.sa31 = SelfAttention(96, 16, heads=2) # -> (B,10,16,16)
        self.sa32 = SelfAttention(96, 16, heads=2) # -> (B,10,16,16)
        self.down3 = Down(96, 288, kernel_size) # -> (B,36,8,8)

        # Bottleneck
        self.bot_conv1 = DoubleConv(288, 288, residual=True) # -> (B,36,8,8)
        self.bot_sa1 = SelfAttention(288, 8, heads=4) # -> (B,36,8,8)
        self.bot_conv2 = DoubleConv(288, 288, residual=True) # -> (B,36,8,8)
        self.bot_sa2 = SelfAttention(288, 8, heads=4) # -> (B,36,8,8)
        self.bot_conv3 = DoubleConv(288, 288, residual=True) # -> (B,36,8,8)
        self.bot_sa3 = SelfAttention(288, 8, heads=4) # -> (B,36,8,8)

        self.up4 = Up(288, 96, skip_channels=96, kernel_size=kernel_size) # -> (B,10,16,16)
        # self.conv4 = DoubleConv(10, 10, residual=True) # -> (B,10,16,16)
        self.sa41 = SelfAttention(96, 16, heads=2) # -> (B,10,16,16)
        self.sa42 = SelfAttention(96, 16, heads=2) # -> (B,10,16,16)

        self.up5 = Up(96, 48, 48, kernel_size=kernel_size) # -> (B,5,24,24)
        # self.conv5 = DoubleConv(5, 5, residual=True) # -> (B,5,24,24)
        self.sa51 = SelfAttention(48, 24, heads=1) # -> (B,5,24,24)
        self.sa52 = SelfAttention(48, 24, heads=1) # -> (B,5,24,24)

        self.up6 = Up(48, 32, skip_channels=32, kernel_size=kernel_size) # -> (B,3,32,32)
        self.sa61 = SelfAttention(32, 32, heads=1) # -> (B,3,32,32)
        self.sa62 = SelfAttention(32, 32, heads=1) # -> (B,3,32,32)

        self.conv_out = nn.Conv2d(32, 3, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (pt.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = pt.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = pt.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = pt.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x:Tensor, t:Tensor) -> Tensor:
        t = t.unsqueeze(-1).type(pt.float)
        t = self.pos_encoding(t, self.time_dim) # (B, time_dim)

        x1 = self.conv_in(x)
        x1 = self.sa11(x1) # -> (B,3,32,32)
        x1 = self.sa12(x1) # -> (B,3,32,32)
        x2 = self.down1(x1, t) # -> (B,5,24,24)

        x2 = self.sa21(x2) # -> (B,5,24,24)
        x2 = self.sa22(x2) # -> (B,5,24,24)
        x3 = self.down2(x2, t) # -> (B,10,16,16)

        x3 = self.sa31(x3) # -> (B,10,16,16)
        x3 = self.sa32(x3) # -> (B,10,16,16)
        x4 = self.down3(x3, t) # -> (B,36,8,8)

        x4 = self.bot_conv1(x4) # -> (B,36,8,8)
        x4 = self.bot_sa1(x4) # -> (B,36,8,8)
        x4 = self.bot_conv2(x4) # -> (B,36,8,8)
        x4 = self.bot_sa2(x4) # -> (B,36,8,8)
        x4 = self.bot_conv3(x4) # -> (B,36,8,8)
        x4 = self.bot_sa3(x4) # -> (B,36,8,8)

        x = self.up4(x4, x3, t) # -> (B,10,16,16)
        x = self.sa41(x) # -> (B,10,16,16)
        x = self.sa42(x) # -> (B,10,16,16)

        x = self.up5(x, x2, t) # -> (B,5,24,24)
        x = self.sa51(x) # -> (B,5,24,24)
        x = self.sa52(x) # -> (B,5,24,24)

        x = self.up6(x, x1, t) # -> (B,3,32,32)
        x = self.sa61(x) # -> (B,3,32,32) 
        x = self.sa62(x)# -> (B,3,32,32)
        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    net = UNet32x32_custom(device="cpu")
    # net = UNet_conditional(num_classes=10, device="cpu")
    print(f"Total Parameters: {sum([p.numel() for p in net.parameters()]):,}")
    x = pt.randn(3, 3, 32, 32)
    t = x.new_tensor([500] * x.shape[0]).long()
    print(net(x, t).shape)
    import code; code.interact(local=locals())