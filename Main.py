import torch
import numpy as np
from torchvision import datasets
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
import time
import math
import pandas as pd
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
###Image directory
imgdir='ROOT_IMGDIR'
####
device="cpu"

###Display images for a given VGG output layer
def displayimgsfromtensor(tensor,f,layer):
    tensor=tensor.squeeze(0).detach()
    print(tensor.shape)
    numimgs=(tensor.shape[0])
    tensor=(tensor-tensor.mean())/tensor.std()
    tensor=(tensor-tensor.min())/(tensor.max()-tensor.min())
    tensor=(tensor*255).to(int)
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    axs = axs.flatten()
    step=max(int(numimgs/25),1)
    for i, ax in enumerate(axs):
        print(tensor[i].shape)
        ax.imshow((tensor[i*step]), cmap='gray')
        ax.set_title("featuremap {}".format(i*step))
        ax.axis('off')
    title=str(f)+"\n layer {}".format(layer)
    plt.suptitle(title)
    plt.show()

    return
###Display initial image
def displayimgfromtensor(tensor):
    t=tensor.squeeze(0)
    t=(t*255).permute(1,2,0)
    im=t.detach().numpy().astype(int)
    plt.imshow(im)
    plt.show()
    return
###Dataset and dataloader
class customData(Dataset):
    def __init__(self,root,isTrain):
        self.root=root
    def __len__(self):
        return len(self.root)
    def __getitem__(self, indx):
        ###image file
        imgfile=self.rootdir[indx]
        im=Image.open(os.path.join(self.root,imgfile))
        transform=transforms.ToTensor()
        tensorimg=transform(im).to(device)
        return tensorimg
Data=customData(imgdir,False)
DataLoader=DataLoader(Data,batch_size=1,shuffle=True)
###Custom VGG class
class VGGwithprint(nn.Module):
    def __init__(self,vgg):
        super(VGGwithprint,self).__init__()
        self.features=vgg.features
        self.avgpool=vgg.avgpool
        self.classifier=vgg.classifier
    def forward(self,x):
        for (i,f) in enumerate(self.features):
            x=f(x)
            displayimgsfromtensor(x,f,i)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
###Model initialization
vgg=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
customvgg=VGGwithprint(vgg)
###Iterate through images and print output images at each layer (for each image in the directory)
def iterateimages():
    for i, (tensor, labels,filenum) in enumerate(DataLoader):
        displayimgfromtensor(tensor)
        logits=customvgg.forward(tensor)
###Call iterateimages
iterateimages()
