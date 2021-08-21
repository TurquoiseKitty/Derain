import torch
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from dataset import show_multi_image, show_tensor_image

def single_forward(fake_image_batch,model,steps,lr=0.05):
    model.zero_grad()
    fake_image_list = []
    for idx in range(steps):
        fake_image_batch = fake_image_batch.detach()
        fake_image_batch.requires_grad_(requires_grad=True)

        energy = model(fake_image_batch)

        im_grad = torch.autograd.grad(energy.sum(),[fake_image_batch],create_graph=True)[0]
        # approx
        # im_grad = im_grad.detach()

        fake_image_batch = fake_image_batch - lr*im_grad
        fake_image_batch = fake_image_batch.clamp(min=-1,max=1)
        fake_image_list.append(fake_image_batch)

    return torch.stack(fake_image_list, dim=0)
'''----some problem here
def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
'''

def train(model, dataloader, steps, epochs, from_model_name="", lr=0.05, optimizer_lr=0.01, demo = False, use_cuda=False, cuda_index=0, save_name = ""):
    print("train model, lr="+str(lr)+", optimizer_lr="+str(optimizer_lr))
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    if from_model_name != "":
        model = torch.load("model_saved/"+from_model_name+".pth")
        model.eval()
    else:
        # model.apply(init_params)
        pass

    if use_cuda:    
        torch.cuda.set_device(cuda_index)
        model = model.cuda(cuda_index)

    for epoch in range(epochs):
        
        sum_loss = 0
        print("epoch : ",epoch)
        for ground_batch, rainy_batch in dataloader:
            if use_cuda:
                ground_batch = ground_batch.cuda(cuda_index)
                rainy_batch = rainy_batch.cuda(cuda_index)

            
            # ---- single forward
            fake_image_batch = single_forward(rainy_batch,model,steps,lr)
            # ---- single forward
            ground_image_batch = ground_batch.unsqueeze(0).repeat(steps,1,1,1,1)

            loss = criterion(fake_image_batch,ground_image_batch)
            losses.append(loss.detach().cpu().numpy())
            sum_loss += loss.item()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("loss : ",sum_loss)
        scheduler.step()

    if demo:
        # select one image and show the effect
        ground_batch, rain_batch = next(iter(dataloader))
        fake_img_batch = single_forward(rain_batch,model,steps,lr)
        show_multi_image(ground_batch)
        for bat in fake_img_batch:
            show_multi_image(bat)

    if save_name != "":
        torch.save(model, "model_saved/"+save_name+".pth")

    return losses



                





