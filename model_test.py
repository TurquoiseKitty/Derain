import torch
from dataset import rainy_dataset, show_tensor_image, show_multi_image, show_concat_image
from torch.utils.data import DataLoader
from model_train import single_forward
from ssim import ssim
from psnr import psnr
import numpy as np
import os, re
# this function will generate a model_report

def test(
    model_path,
    dataloader,
    dataset_name,
    rainy_extent,
    lr,
    steps = 5,
    use_cuda=False, 
    cuda_index=0
    ):

    form=re.compile("\S+\."+"pth")
    for filename in os.listdir(model_path):
        if form.match(filename):
            model = torch.load(model_path+"/"+filename)
            model = model.cpu()
            if use_cuda:
                model = model.cuda(cuda_index)
            model.eval()
            break

    losses_ssim = []
    losses_psnr = []
    for ground_batch, rainy_batch in dataloader:
        if use_cuda:
            ground_batch = ground_batch.cuda(cuda_index)
            rainy_batch = rainy_batch.cuda(cuda_index)

        fake_image_output = single_forward(rainy_batch,model,steps,lr)[-1]
        losses_ssim.append(ssim(ground_batch,fake_image_output,window_size=5).detach().cpu().numpy())
        losses_psnr.append(psnr(ground_batch,fake_image_output).detach().cpu().numpy())

    loss_ssim_estimate = np.mean(losses_ssim)
    loss_psnr_estimate = np.mean(losses_psnr)



    file = open(model_path+"/"+"abstract.txt","a")
    file.write("\n")
    file.write("**********************************\n")
    file.write("test model on dataset : "+dataset_name+"\n")
    file.write("    with rainy extent : "+str(rainy_extent)+"\n")
    file.write("steps of inner loop : "+str(steps)+"\n")
    file.write("learning rate for inner loop : "+str(lr)+"\n")
    file.write("ssim loss : "+str(loss_ssim_estimate)+"\n")
    file.write("psnr loss : "+str(loss_psnr_estimate)+"\n")
    file.close()
    print("finish evaluation")

def demo(
    model_path,
    ground_image,
    rainy_image,
    lr,
    steps = 5,
    demo_name=""
    ):
    form=re.compile("\S+\."+"pth")
    for filename in os.listdir(model_path):
        if form.match(filename):
            model = torch.load(model_path+"/"+filename)
            model = model.cpu()
            model.eval()
            break

    img_list = [ground_image.detach().cpu(),rainy_image.detach().cpu()]
    rainy_batch = rainy_image.unsqueeze(0)
    process = single_forward(rainy_batch,model,steps,lr)
    for img in process:
        img_list.append(img[0].detach().cpu())

    img_dir = model_path+"/"+"practical_demo"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    show_concat_image(img_list,show=False,save=True,save_folder=img_dir,save_name="demo_"+demo_name)

if __name__ == "__main__":
    testing_dataset=rainy_dataset(data_path="large_datasets/rainy_image_dataset",data_subpath="testing",data_len=3, rainy_extent=10,normalize=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size = 1, shuffle=True)
    count = 0
    for ground_batch, rainy_batch in testing_dataloader:    
        ground_image = ground_batch[0]
        rainy_image = rainy_batch[0]

        demo(
            "model_bundle/model_GroupNorm_lamCD05_lr5_optlr001",
            ground_image,
            rainy_image,
            0.5,
            steps = 5,
            demo_name="demo"+str(count)
        )
        demo(
            "model_bundle/model_GroupNorm_lamCD05_lr1_optlr001",
            ground_image,
            rainy_image,
            0.5,
            steps = 5,
            demo_name="demo"+str(count)
        )
        demo(
            "model_bundle/model_GroupNorm_lamCD05_lr2_optlr001",
            ground_image,
            rainy_image,
            0.5,
            steps = 5,
            demo_name="demo"+str(count)
        )
        
        count += 1