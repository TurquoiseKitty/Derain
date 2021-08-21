import torch

def psnr(img1, img2):
    # image to [0,255]
    img1 = (img1 +1)/2*255
    img2 = (img2 +1)/2*255
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255 / torch.sqrt(mse))