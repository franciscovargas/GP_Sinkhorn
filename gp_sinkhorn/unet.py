from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from collections import defaultdict
from gp_sinkhorn.mem_utils import get_size_to_live_tensors, get_tensor_size, print_gpu_mem_usage

from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torchvision
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), verbose=False):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
        self.verbose = verbose
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            if self.verbose:
                print(f"enc1: {x.shape}")
            x = block(x)
            if self.verbose:
                print(f"enc2: {x.shape}")
            ftrs.append(x)
            x = self.pool(x)
            if self.verbose:
                print(f"enc3: {x.shape}")
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), verbose=False):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        self.verbose = verbose
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            if self.verbose:
                print(f"dec1: {x.shape}")
            x        = self.upconvs[i](x)
            if self.verbose:
                print(f"dec2: {x.shape}")
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            if self.verbose:
                print(f"dec3: {x.shape}")            
            x        = self.dec_blocks[i](x)
            if self.verbose:
                print(f"dec4: {x.shape}")            
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), 
                 dec_chs=(1024, 512, 256, 128, 64), num_class=1, 
                 retain_dim=False, out_sz=(28, 28),
                 verbose=False):
        super().__init__()
        self.encoder     = Encoder(enc_chs, verbose=verbose)
        self.decoder     = Decoder(dec_chs, verbose=verbose)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz = out_sz
        self.verbose = verbose


    def forward(self, x):
        if self.verbose:
            print(f"f1: {x.shape}")          
        enc_ftrs = self.encoder(x)
        if self.verbose:
            print(f"f2: {enc_ftrs[::-1][0].shape}")          
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        if self.verbose:
            print(f"f3: {x.shape}")           

        out      = self.head(out)
        if self.verbose:
            print(f"f4: {x.shape}")           
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
            if self.verbose:
                print(f"f5: {x.shape}")      
        return out
    
    
def get_trained_unet(Xs, Ys, device, num_epochs=50, batch_size=None):
    
    with torch.no_grad():
        # I *think* it's legitimate to put this all in a no_grad()...
        num_samples = Xs.size(0)
        Xs = Xs[:, :-1]
        assert Ys.size(0) == num_samples
        xs_reshaped = Xs.reshape(num_samples, 1, 28, 28)
        ys_reshaped = Ys.reshape(num_samples, 1, 28, 28)
        unet = UNet(enc_chs=(1, 64, 128), dec_chs=(128, 64), retain_dim=True).double().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(unet.parameters(), lr=0.001)

        if batch_size is None:
            batch_size = xs_reshaped.size(0)
        ds = TensorDataset(xs_reshaped, ys_reshaped)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    losses = []
    for epoch in range(num_epochs): 
        for i, data in enumerate(dl):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = unet(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
#         losses.append(loss.item())
#     plt.plot(list(range(len(losses))), losses)
#     plt.show()
    return lambda x: unet(x[:, :-1].reshape(-1, 1, 28, 28)).reshape(-1, 784)


