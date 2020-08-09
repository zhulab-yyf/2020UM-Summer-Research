from __future__ import absolute_import, division, print_function
import torch
import torchvision
import pickle as pkl
from torch import nn
import torch.nn.functional as F


class DRIONSNet(nn.Module):
    def __init__(self):
        super(DRIONSNet, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)

        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)

        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode = True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv2_2_16 = nn.Conv2d(128, 16, 3, 1, 1)
        self.conv3_3_16 = nn.Conv2d(256, 16, 3, 1, 1)
        self.conv4_3_16 = nn.Conv2d(512, 16, 3, 1, 1)
        self.conv5_3_16 = nn.Conv2d(512, 16, 3, 1, 1)

        self.upsample2 = nn.ConvTranspose2d(16, 16, 4, 2, 1)
        self.upsample4 = nn.ConvTranspose2d(16, 16, 8, 4, 0)
        self.upsample8 = nn.ConvTranspose2d(16, 16, 16, 8, 0)
        self.upsample16 = nn.ConvTranspose2d(16, 16, 32, 16, 0)

        self.new_score_weighting = nn.Conv2d(64, 1, 1)


    def forward(self, inputs):
        x = F.relu(self.conv1_1(inputs))
        x = F.relu(self.conv1_2(x))

        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        tempx1 = F.relu(self.conv2_2(x))


        x = self.pool2(tempx1)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        tempx2 = F.relu(self.conv3_3(x))
        
        
        x = self.pool3(tempx2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        tempx3 = F.relu(self.conv4_3(x))
        
        x = self.pool4(tempx3)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        
        
        tempx1 = self.conv2_2_16(tempx1)
        tempx1 = self.upsample2(tempx1)
        tempx2 = self.upsample4(self.conv3_3_16(tempx2))
        tempx3 = self.upsample8(self.conv4_3_16(tempx3))
        x = self.upsample16(self.conv5_3_16(x))
     #   print(tempx1.shape)

        tempx2 = tempx2[:, :, 0:400,  0:600]
        tempx3 = tempx3[:, :, 0:400,  0:600]
        x = x[:, :, 0:400, 0:600]

        tempx1 = torch.cat((tempx1, tempx2), 1)
        x = torch.cat((tempx3, x), 1)
        x = torch.cat((tempx1, x), 1)

        x = self.new_score_weighting(x)
        x = torch.sigmoid(x)

        return x

    def load_weights_from_pkl(self, weights_pkl):
        from torch import from_numpy
        with open(weights_pkl, 'rb') as wp:
            try:
                # for python3
                name_weights = pkl.load(wp, encoding='latin1')
            except TypeError as e:
                # for python2
                name_weights = pkl.load(wp)
            state_dict = {}
            def _set(layer, key):
                state_dict[layer + '.weight'] = from_numpy(name_weights[key]['weight'])
                state_dict[layer + '.bias'] = from_numpy(name_weights[key]['bias'])

            _set('conv1_1', 'conv1_1')
            _set('conv1_2', 'conv1_2')

            _set('conv2_1', 'conv2_1')
            _set('conv2_2', 'conv2_2')

            _set('conv3_1', 'conv3_1')
            _set('conv3_2', 'conv3_2')
            _set('conv3_3', 'conv3_3')

            _set('conv4_1', 'conv4_1')
            _set('conv4_2', 'conv4_2')
            _set('conv4_3', 'conv4_3')

            _set('conv5_1', 'conv5_1')
            _set('conv5_2', 'conv5_2')
            _set('conv5_3', 'conv5_3')

            _set('conv2_2_16', 'conv2_2_16')
            _set('conv3_3_16', 'conv3_3_16')
            _set('conv4_3_16', 'conv4_3_16')
            _set('conv5_3_16', 'conv5_3_16')

            _set('upsample2', 'upsample2_')
            _set('upsample4', 'upsample4_')
            _set('upsample8', 'upsample8_')
            _set('upsample16', 'upsample16_')

            _set('new_score_weighting', 'new-score-weighting')
            self.load_state_dict(state_dict)


if __name__ == '__main__':
    net = DRIONSNet()
    net.eval()

    print(len(list(net.named_parameters())))
    for name, param in list(net.named_parameters()):
        print(name, param.size())
