import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from models.model import UNet, MiniUNet
from datasets.dataset import TabletDataset
from torch.utils.data import DataLoader
import torchvision.models as models

from glob import glob


class generator:
    def __init__(self, in_channels, out_channels, device, use_cuda=False):
        self.generatorNet = UNet(${1:n_channels}, ${2:n_classes}, ${3:bilinear})$0(in_channels, out_channels, True)
        self.discriminatorNet = models.vgg16_bn()
        if use_cuda:
            self.generatorNet = self.generatorNet.cuda(device)
            self.discriminatorNet = self.discriminatorNet.cuda(device)

        self.optimizer = optim.SGD(self.generatorNet.parameters(),
                                   lr=0.02, momentum=0)
        self.device = device

        if use_cuda:
            print("model:{}\nmemory consumption:{}MB\ncache consumption:{}MB".format(
                self.generatorNet.__class__.__name__, torch.cuda.memory_allocated(
                    self.device)//2**20,
                torch.cuda.memory_cached(self.device)//2**20))

    def train(self, dataset, epochs=10, batch_size=1, shuffle=True, num_workers=0, modelpth="."):
        print("Dataset size:{}".format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size,
                                shuffle, num_workers=num_workers, drop_last=True)
        training_start = datetime.now()
        for i in range(epochs):
            print("epoch:{}".format(i))
            sum_loss = 0
            epoch_start = datetime.now()
            for x, y in dataloader:
                #  print(x[0,:,100:200, 200:400].cpu().numpy())
                self.optimizer.zero_grad()
                output = self.generatorNet(
                    x.reshape(batch_size, 1, x.shape[1], x.shape[2]).to(self.device))
                target = y.to(self.device)
                criterion = nn.MSELoss()
                output = output.view(batch_size, -1)
                target = target.contiguous().view(batch_size, -1)
                loss = criterion(output, target)
                sum_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            # TODO: consider last cut
            print("Epoch loss:{}".format(sum_loss/len(dataset)*batch_size))
            print("Epoch Time:{}".format(datetime.now()-epoch_start))
            torch.save(self.generatorNet.state_dict(), modelpth)
        print("Total Elapsed Time:{}".format(datetime.now()-training_start))

    def generate(self, image):
        with torch.no_grad():
            x = torch.from_numpy(image.reshape(
                1, 1, image.shape[0], image.shape[1])).to(self.device)
            output = self.generatorNet(x)
        return output


if __name__ == "__main__":

    #  print(glob('../images/*'))

    DATASET = TabletDataset(
        '../images', '../line_images', crop_size=(500, 500))

    USE_CUDA = bool(torch.cuda.is_available())

    device0 = torch.device("cuda:0" if USE_CUDA else "cpu")
    model = generator(1, 1, device=device0, use_cuda=USE_CUDA)
    model.train(DATASET, epochs=5, batch_size=5,
                modelpth="../model_weights/unet-canny-10-20.pth")
