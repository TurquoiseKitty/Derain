import torch
import torch.optim as optim
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from dataset import show_multi_image, show_tensor_image, show_concat_image
from misc import AverageMeter, MovingAverage
from PIL import Image
import numpy as np
import os, sys, pdb
from model import ResNetModel

def single_forward(fake_image_batch,model,steps,lr=1):
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

# each model will be stored in separate folder
# all analysis related to the model will be in this folder
def train(dataloader, dataset_name, rainy_extent, epochs, model_name, save_path="model_bundle", from_model_path="", use_cuda=True, cuda_index=0, \
                            args={'steps':5,'lr':0.1,'optimizer_lr':0.001,'lambda_cd':0.01,'lambda_reg':1e-4}):
    lr = args['lr']
    optimizer_lr = args['optimizer_lr']
    steps = args['steps']
    lambda_cd = args['lambda_cd']
    lambda_reg = args['lambda_reg']

    directory = save_path+"/"+model_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if from_model_path != "":
        model = torch.load(from_model_path)
        model.eval()
    else:
        model = ResNetModel(args = None)
        # model.apply(init_params)
        pass

    print("training model : "+model_name)
    print("lr : "+str(lr)+", opti_lr : "+str(optimizer_lr)+", lam_cd : "+str(lambda_cd)+", lam_reg : "+str(lambda_reg))
    # create a basic info
    file = open(directory+"/abstract.txt","w")
    file.write("model name : "+model_name+"\n")
    file.write("trained on dataset : "+dataset_name+"\n")
    file.write("    with rainy extent : "+str(rainy_extent)+"\n")
    file.write("steps of inner loop : "+str(steps)+"\n")
    file.write("learning rate for inner loop : "+str(lr)+"\n")
    file.write("optimizer learning rate : "+str(optimizer_lr)+"\n")
    file.write("lambda for cd loss : "+str(lambda_cd)+"\n")
    file.write("lambda for reg loss : "+str(lambda_reg)+"\n")
    file.write("\n")
    file.write(str(epochs)+" epochs are set \n")
    file.close()

    losses = []
    mse_losses = []
    positive_energys = []
    negative_energys = []
    cd_losses = []

    

    loss_ma = MovingAverage(20)
    eng_pos_ma = MovingAverage(20)
    eng_neg_ma = MovingAverage(20)
    cd_ma = MovingAverage(20)
    mse_ma = MovingAverage(20)

    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.MSELoss()
    
    

    if use_cuda:    
        torch.cuda.set_device(cuda_index)
        model = model.cuda(cuda_index)

    mul_fac = 100.0

    for epoch in range(epochs):
        
        sum_loss = 0
        print("epoch : ",epoch)
        for ground_batch, rainy_batch in dataloader:
            if use_cuda:
                ground_batch = ground_batch.cuda(cuda_index)
                rainy_batch = rainy_batch.cuda(cuda_index)

            
            # ---- single forward
            fake_image_batch = single_forward(rainy_batch,model,steps,lr)

            positive_energy = model(ground_batch.detach()).sum()
            negative_energy = model(fake_image_batch[-1].detach()).sum()


            # ---- single forward
            ground_image_batch = ground_batch.unsqueeze(0).repeat(steps,1,1,1,1)

            loss_mse = criterion(fake_image_batch*mul_fac, ground_image_batch*mul_fac)
            loss_cd = positive_energy - negative_energy
            loss = loss_mse + lambda_cd * loss_cd + lambda_reg * (positive_energy ** 2 + negative_energy ** 2)
            

            
            sum_loss += loss.item()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            mse_ma.update(loss_mse.item())
            eng_pos_ma.update(positive_energy.item())
            eng_neg_ma.update(negative_energy.item())
            cd_ma.update(loss_cd.item())
            loss_ma.update(loss.item())

            losses.append(float(loss_ma.avg))
            mse_losses.append(float(mse_ma.avg))
            positive_energys.append(float(eng_pos_ma.avg))
            negative_energys.append(float(eng_neg_ma.avg))
            cd_losses.append(float(cd_ma.avg))
            

        img_list = [img.detach().cpu() for img in fake_image_batch.permute(1,0,2,3,4)[0]]
        img_list.insert(0,rainy_batch[0].detach().cpu())
        img_list.insert(0,ground_batch[0].detach().cpu())

        img_dir = directory+"/"+"img_sample_demo"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        show_concat_image(img_list,show=False,save=True,save_folder=img_dir,save_name="demo_epoch_"+str(epoch))
        print("loss : %.2f | pos eng: %.2f, neg eng: %.2f | cd: %.2f | mse: %.2f" \
            %(loss_ma.avg, eng_pos_ma.avg, eng_neg_ma.avg, cd_ma.avg, mse_ma.avg))
        # scheduler.step()


    
    torch.save(model, directory+"/"+model_name+".pth")

    return mse_losses, positive_energys, negative_energys,cd_losses, losses


