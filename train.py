import torch
import torch.optim as optim
import torch.nn as nn

def single_forward(fake_image,model,steps,lr=0.05):

    fake_image_list = []

    for idx in range(steps):
        fake_image = fake_image.detach()
        fake_image.requires_grad_(requires_grad=True)

        energy = model(fake_image.unsqueeze(0))[0]

        im_grad = torch.autograd.grad(energy,[fake_image],create_graph=True)[0]
        # approx
        im_grad = im_grad.detach()

        fake_image = fake_image - lr*im_grad
        fake_image = fake_image.clamp(min=-1,max=1)
        fake_image_list.append(fake_image)

    return fake_image_list



def train(model, dataloader, steps, epochs, lr=0.05, optimizer_lr=0.001):

    optimizer = optim.SGD(model.parameters(), lr=optimizer_lr, momentum=0.9)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print("epoch : ",epoch)
        for image, fake_image in dataloader:
            optimizer.zero_grad()
            # ---- single forward
            # fake_image_list = single_forward(fake_img,model,steps,lr)
            fake_image_list = []

            for idx in range(steps):
                fake_image = fake_image.detach()
                fake_image.requires_grad_(requires_grad=True)

                energy = model(fake_image.unsqueeze(0))[0]

                im_grad = torch.autograd.grad(energy,[fake_image],create_graph=True)[0]
                # approx
                im_grad = im_grad.detach()

                fake_image = fake_image - lr*im_grad
                fake_image = fake_image.clamp(min=-1,max=1)
                fake_image_list.append(fake_image)
            # ---- single forward
            for img_gen in fake_image_list:
                loss = criterion(img_gen, image)
                print(loss.item())
                loss.backward()
                optimizer.step()





