import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from models.model import UNet, one_color_resnet
from datasets.dataset import TabletDataset
from torch.utils.data import DataLoader

from glob import glob

from tensorboardX import SummaryWriter


class generator:
    def __init__(self, in_channels, out_channels, device, use_cuda=False):
        self.generatorNet = UNet(in_channels, out_channels, True)
        self.discriminatorNet = one_color_resnet(2, 1)

        self.use_multiple_gpu = isinstance(device, tuple)

        if use_cuda:
            if self.use_multiple_gpu:
                self.generatorNet = self.generatorNet.cuda(device[0])
                self.discriminatorNet = self.discriminatorNet.cuda(device[1])
            else:
                self.generatorNet = self.generatorNet.cuda(device)
                self.discriminatorNet = self.discriminatorNet.cuda(device)

        self.optimizer_G = optim.SGD(self.generatorNet.parameters(),
                                     lr=0.2, momentum=0.99)

        self.optimizer_D = optim.SGD(self.discriminatorNet.parameters(),
                                     lr=0.02, momentum=0.99)
        self.device = device

        self.writer = SummaryWriter()

        #  if use_cuda:
        #      print("model:{}\nmemory consumption:{}MB\ncache consumption:{}MB".format(
        #          self.generatorNet.__class__.__name__, torch.cuda.memory_allocated(
        #              self.device)//2**20,
        #          torch.cuda.memory_cached(self.device)//2**20))

    def train(self, dataset, epochs=10, batch_size=1, shuffle=True, num_workers=0, modelpth="."):
        print("Dataset size:{}".format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size,
                                shuffle, num_workers=num_workers, drop_last=True)
        training_start = datetime.now()
        if self.use_multiple_gpu:
            for i in range(epochs):
                print("epoch:{}".format(i))
                sum_g_loss = 0
                sum_fake_loss = 0
                sum_real_loss = 0
                epoch_start = datetime.now()
                for x, ytrue in dataloader:
                    #  print(x[0,:,100:200, 200:400].cpu().numpy())
                    x = x.reshape(batch_size, 1, x.shape[1], x.shape[2])
                    ytrue = ytrue.reshape(
                        batch_size, 1, ytrue.shape[1], ytrue.shape[2])
                    yfake = self.generatorNet(x.to(self.device[0]))
                    #  criterion = nn.MSELoss()
                    #  yfake = yfake.view(batch_size, -1)
                    #  ytrue = ytrue.contiguous().view(batch_size, -1)
                    #  loss = criterion(yfake, ytrue)
                    prediction = self.discriminatorNet(
                        torch.cat((yfake.to(self.device[1]), x.to(self.device[1])), 1))
                    criterion = nn.BCELoss()
                    g_loss = criterion(prediction, torch.tensor(
                        [0.]*batch_size).to(self.device[1]))
                    self.optimizer_G.zero_grad()
                    g_loss.backward()
                    #  self.optimizer_G.step()

                    prediction = self.discriminatorNet(
                        torch.cat((yfake.detach().to(self.device[1]), x.to(self.device[1])), 1))
                    fake_loss = - \
                        criterion(prediction, torch.tensor(
                            [0.]*batch_size).to(self.device[1]))
                    self.optimizer_D.zero_grad()
                    fake_loss.backward()
                    self.optimizer_D.step()

                    ytrue = ytrue.to(self.device[1])
                    #  print(ytrue.shape)
                    #  print(yfake.shape)
                    prediction = self.discriminatorNet(
                        torch.cat((ytrue.to(self.device[1]), x.to(self.device[1])), 1))
                    real_loss = criterion(prediction, torch.tensor(
                        [0.]*batch_size).to(self.device[1]))
                    self.optimizer_D.zero_grad()
                    real_loss.backward()
                    self.optimizer_D.step()

                    sum_g_loss += g_loss.item()
                    sum_real_loss += real_loss.item()
                    sum_fake_loss -= fake_loss.item()

                for name, param in self.discriminatorNet.named_parameters():
                    self.writer.add_histogram(
                        name, param.clone().cpu().data.numpy(), i)

                self.writer.add_scalar('data/g_loss', sum_g_loss, i)
                self.writer.add_scalar('data/real_loss', sum_real_loss, i)
                self.writer.add_scalar('data/fake_loss', sum_fake_loss, i)

                #  self.writer.export_scalars_to_json("./all_scalars.json")

                # TODO: consider last cut
                print("Epoch gene loss:{}".format(
                    sum_g_loss/len(dataset)*batch_size))
                print("Epoch fake loss:{}".format(
                    sum_fake_loss/len(dataset)*batch_size))
                print("Epoch real loss:{}".format(
                    sum_real_loss/len(dataset)*batch_size))
                print("Epoch Time:{}".format(datetime.now()-epoch_start))
                torch.save(self.generatorNet.state_dict(), modelpth)
            print("Total Elapsed Time:{}".format(
                datetime.now()-training_start))

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
    device1 = torch.device("cuda:1" if USE_CUDA else "cpu")
    model = generator(1, 1, device=(device0, device1), use_cuda=USE_CUDA)
    model.train(DATASET, epochs=100, batch_size=5,
                modelpth="../model_weights/unet-canny-10-20.pth")
