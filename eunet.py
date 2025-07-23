import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果用多张卡也一样
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FlowDenoiseDataset(Dataset):
    def __init__(self, noisy_path, clean_path):
        self.noisy = np.load(noisy_path)  # [N, 2, H, W]
        self.clean = np.load(clean_path)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        x = torch.tensor(self.noisy[idx], dtype=torch.float32)
        y = torch.tensor(self.clean[idx], dtype=torch.float32)
        return x, y

class DoubleConv(nn.Module):
    """Conv2d + BatchNorm2d + ReLU（×2）"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

def crop_to_fit(src, target):
    _, _, h_src, w_src = src.size()
    _, _, h_tgt, w_tgt = target.size()
    dh = (h_src - h_tgt) // 2
    dw = (w_src - w_tgt) // 2
    return src[:, :, dh:dh + h_tgt, dw:dw + w_tgt]

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(UNet, self).__init__()

        # 编码器
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # 中间层
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d1 = self.up1(b)
        e4_cropped = crop_to_fit(e4, d1)
        d1 = self.dec1(torch.cat([d1, e4_cropped], dim=1))

        d2 = self.up2(d1)
        e3_cropped = crop_to_fit(e3, d2)
        d2 = self.dec2(torch.cat([d2, e3_cropped], dim=1))

        d3 = self.up3(d2)
        e2_cropped = crop_to_fit(e2, d3)
        d3 = self.dec3(torch.cat([d3, e2_cropped], dim=1))

        d4 = self.up4(d3)
        e1_cropped = crop_to_fit(e1, d4)
        d4 = self.dec4(torch.cat([d4, e1_cropped], dim=1))

        out = self.out_conv(d4)

        # 强制插值到输入大小
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        
        # 编码器 4层
        self.enc1 = DoubleConv(in_channels, 64)
        self.res1 = ResidualBlock(64)
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        self.enc2 = DoubleConv(128, 128)
        self.res2 = ResidualBlock(128)
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        self.enc3 = DoubleConv(256, 256)
        self.res3 = ResidualBlock(256)
        self.down3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        
        self.enc4 = DoubleConv(512, 512)
        self.res4 = ResidualBlock(512)
        self.down4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        
        # Bottleneck
        self.middle = DoubleConv(1024, 1024)
        self.res_middle = ResidualBlock(1024)
        
        # 解码器 4层，上采样用Upsample+卷积
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dec1 = DoubleConv(1024, 512)
        self.res_dec1 = ResidualBlock(512)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec2 = DoubleConv(512, 256)
        self.res_dec2 = ResidualBlock(256)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = DoubleConv(256, 128)
        self.res_dec3 = ResidualBlock(128)
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec4 = DoubleConv(128, 64)
        self.res_dec4 = ResidualBlock(64)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e1 = self.res1(e1)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        e2 = self.res2(e2)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        e3 = self.res3(e3)
        d3 = self.down3(e3)

        e4 = self.enc4(d3)
        e4 = self.res4(e4)
        d4 = self.down4(e4)

        m = self.middle(d4)
        m = self.res_middle(m)

        u1 = self.up1(m)
        u1 = crop_to_fit(u1, e4)  # 裁剪u1使其和e4尺寸对齐
        u1 = torch.cat([u1, e4], dim=1)
        u1 = self.dec1(u1)
        u1 = self.res_dec1(u1)

        u2 = self.up2(u1)
        u2 = crop_to_fit(u2, e3)
        u2 = torch.cat([u2, e3], dim=1)
        u2 = self.dec2(u2)
        u2 = self.res_dec2(u2)

        u3 = self.up3(u2)
        u3 = crop_to_fit(u3, e2)
        u3 = torch.cat([u3, e2], dim=1)
        u3 = self.dec3(u3)
        u3 = self.res_dec3(u3)

        u4 = self.up4(u3)
        u4 = crop_to_fit(u4, e1)
        u4 = torch.cat([u4, e1], dim=1)
        u4 = self.dec4(u4)
        u4 = self.res_dec4(u4)

        return self.out(u4)

def sobel_edge(x):
    """
    Compute Sobel edge map for input tensor x. This is differentiable w.r.t. x.
    x: (B, C, H, W)
    returns: edge magnitude map (B, C, H, W)
    """
    device = x.device
    sobel_kernel_x = torch.tensor([[1., 0., -1.],
                                   [2., 0., -2.],
                                   [1., 0., -1.]], device=device).reshape(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[1., 2., 1.],
                                   [0., 0., 0.],
                                   [-1., -2., -1.]], device=device).reshape(1, 1, 3, 3)
    
    # Repeat to apply on all channels independently
    C = x.shape[1]
    weight_x = sobel_kernel_x.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
    weight_y = sobel_kernel_y.repeat(C, 1, 1, 1)

    # groups=C ensures per-channel convolution
    grad_x = F.conv2d(x, weight_x, padding=1, groups=C)
    grad_y = F.conv2d(x, weight_y, padding=1, groups=C)

    edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    return edge

def edge_loss(pred, target):
    pred_edge = sobel_edge(pred)
    target_edge = sobel_edge(target)
    return F.l1_loss(pred_edge, target_edge)

# 训练时示例loss计算
def compute_loss(output, target):
    mse = F.mse_loss(output, target)
    eloss = edge_loss(output, target)
    total_loss = mse + 0.1 * eloss  # 可以调节权重0.1
    # total_loss = mse
    return total_loss * 1e3, mse, eloss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数
batch_size = 8
epochs = 500
lr = 1e-3

# 数据加载
dataset = FlowDenoiseDataset('data/train_noisy.npy', 'data/train_clean.npy')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = FlowDenoiseDataset('data/valid_noisy.npy', 'data/valid_clean.npy')
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型
model = ImprovedUNet().to(device)
# model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

set_seed(1337)

best_val_loss = 1e9
# 训练
for epoch in range(epochs):
    model.train()
    epoch_start_time = time.time() # 在每个 epoch 开始时记录时间
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss, mse_loss, edge_l = compute_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    epoch_end_time = time.time() # 在每个 epoch 结束时记录时间
    epoch_duration = epoch_end_time - epoch_start_time # 计算当前 epoch 的耗时

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}, Time: {epoch_duration:.2f}s", flush=True)
    # 每10轮验证
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_val, y_val in val_dataloader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_pred = model(x_val)
                val_loss, _, _ = compute_loss(val_pred, y_val)
                val_loss_total += val_loss.item()
        
        avg_val_loss = val_loss_total / len(val_dataloader)

        # 如果验证集loss更低，保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model_file/our_3ma_wodec2.pth')
            print(f"New best model saved with val loss {best_val_loss:.6f}", flush=True)

# model.load_state_dict(torch.load('model_file/unet_denoise_10km.pth'))
# test_flow = np.load("test_data/noisy_flow_city.npy")  # shape: (1, 2, H, W)
# x = torch.tensor(test_flow, dtype=torch.float32).to(device)
# model.eval()
# with torch.no_grad():
#     denoised = model(x)[0].cpu().numpy()  # (1, 2, H, W)
# np.save("denoised_flow_pred.npy", denoised)