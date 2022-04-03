import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchsummary import summary


class type1(nn.Module):
    def __init__(self,input_channel, output_channel):
        super(type1, self).__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.output_channel = output_channel

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class type2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(type2, self).__init__()
        self.layer1 = type1(input_channel,output_channel)
        self.input_channel = self.layer1.output_channel

        self.conv1 = nn.Conv2d(self.input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)

    def forward(self,x):
        out = self.layer1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = out + x
        return out

class type3(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(type3, self).__init__()
        self.layer1 = type1(input_channel, output_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.branch_conv = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=2)
        self.branch_bn = nn.BatchNorm2d(output_channel)

    def forward(self,x):
        out1 = self.layer1(x)
        out1 = self.conv1(out1)
        out1 = self.bn1(out1)
        out1 = self.avg_pool(out1)

        out2 = self.branch_conv(x)
        out2 = self.branch_bn(out2)

        return out1+out2


class type4(nn.Module):
    def __init__(self,intput_channel, output_channel):
        super(type4, self).__init__()
        self.layer1 = type1(intput_channel, output_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = self.layer1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = torch.mean(out, dim=(2,3), keepdim=True)
        out = out.view(out.shape[0],1, -1)
        out = torch.squeeze(out)
        return out

class SRNet(nn.Module):
    def __init__(self, input_size):
        super(SRNet, self).__init__()
        #第一部分
        self.layer1 = type1(input_channel=input_size, output_channel=64)
        self.layer2 = type1(input_channel=64, output_channel=16)
        #第二部分
        self.layer3 = type2(input_channel=16, output_channel=16)
        self.layer4 = type2(input_channel=16, output_channel=16)
        self.layer5 = type2(input_channel=16, output_channel=16)
        self.layer6 = type2(input_channel=16, output_channel=16)
        self.layer7 = type2(input_channel=16, output_channel=16)
        #第三部分
        self.layer8 = type3(input_channel=16, output_channel=16)
        self.layer9 = type3(input_channel=16, output_channel=64)
        self.layer10 = type3(input_channel=64, output_channel=128)
        self.layer11 = type3(input_channel=128, output_channel=256)
        #第四部分
        self.layer12 = type4(intput_channel=256, output_channel=512)
        #全连接层
        self.FC = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)

        out = self.FC(out)
        return out



if __name__ == '__main__':
    model = SRNet(1)
    # summary(model, input_size=[(1, 256, 256)], batch_size=4, device='cpu')

    for name,p in model.named_parameters():
        if ('conv' in name) and ('weight' in name):
            print(name)

    print("")

    for name,p in model.named_parameters():
        if ('conv' not in name) or ('weight' not in name):
            print(name)

    # g = [
    #     {'params': (p for name, p in model.named_parameters() if 'bias' not in name), 'weight_decay': 0.0001},
    #     {'params': (p for name, p in model.named_parameters() if 'bias' in name)}]
    #
    # optimizer = torch.optim.Adam(g , lr=0.01)
    #
    # print(optimizer)


