"""Simple, well-tested 3D U-Net implementation.
Designed to be readable and reasonably memory-efficient.
"""


import torch
import torch.nn as nn




class DoubleConv(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.net = nn.Sequential(
    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm3d(out_ch),
    nn.ReLU(inplace=True),
    nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm3d(out_ch),
    nn.ReLU(inplace=True),
    )
    
  
  def forward(self, x):
    return self.net(x)




class Down(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.pool = nn.MaxPool3d(2)
    self.conv = DoubleConv(in_ch, out_ch)
  
  
  def forward(self, x):
    return self.conv(self.pool(x))




class Up(nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    # use ConvTranspose for upsampling
    self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_ch, out_ch)
    
  
  def forward(self, x1, x2):
    x1 = self.up(x1)
    # pad if necessary
    diffZ = x2.size(2) - x1.size(2)
    diffY = x2.size(3) - x1.size(3)
    diffX = x2.size(4) - x1.size(4)
    x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
    diffY // 2, diffY - diffY // 2,
    diffZ // 2, diffZ - diffZ // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)




class UNet3D(nn.Module):
  def __init__(self, in_channels=1, out_channels=1, base_filters=16):
    super().__init__()
    self.inc = DoubleConv(in_channels, base_filters)
    self.down1 = Down(base_filters, base_filters * 2)
    self.down2 = Down(base_filters * 2, base_filters * 4)
    self.down3 = Down(base_filters * 4, base_filters * 8)
    
    
    self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)
    
    
    self.up3 = Up(base_filters * 16, base_filters * 8)
    self.up2 = Up(base_filters * 8, base_filters * 4)
    self.up1 = Up(base_filters * 4, base_filters * 2)
    self.up0 = Up(base_filters * 2, base_filters)
    
    
    self.out_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
  
  
  def forward(self, x):
    x0 = self.inc(x)
    x1 = self.down1(x0)
    x2 = self.down2(x1)
    x3 = self.down3(x2)
    b = self.bottleneck(x3)
    u3 = self.up3(b, x3)
    u2 = self.up2(u3, x2)
    u1 = self.up1(u2, x1)
    u0 = self.up0(u1, x0)
    out = self.out_conv(u0)
    return out



if __name__ == "__main__":
    # quick smoke test
    m = UNet3D(in_channels=1, out_channels=1, base_filters=8)
    x = torch.randn(1, 1, 64, 128, 128)
    y = m(x)
    print('output shape', y.shape)