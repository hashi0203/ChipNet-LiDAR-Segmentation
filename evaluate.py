import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import numpy as np
import os
import argparse

from utility import *
from data_loader import *
from model import ChipNet

parser = argparse.ArgumentParser(description='PyTorch KITTI Point Cloud Data Evaluation')
parser.add_argument('--file', metavar='FILE', type=str, help='ckpt file path')
args = parser.parse_args()

# img_types: image types in dataset
img_types = ['um', 'umm', 'uu']
# datanum: number of images with corresponding img_types (must be list of length 3)
datanum = [1] * 3
# startidx: start index of images with corresponding img_types (must be list of length 3)
startidx = [94] * 3

print('==> Preparing eval data..')
transform = transforms.Compose([transforms.ToTensor()])
evalset = datasetsKitti('data_road', datanum, startidx=startidx, transforms=transform, progress=True)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Loading checkpoint..')
checkpoint = torch.load(args.file)
isMSE = (checkpoint['criterion'] == 'MSE')

if isMSE:
    net = ChipNet(och=1).to(device)
else:
    net = ChipNet().to(device)

if device == 'cuda':
    # Acceleration by using DataParallel
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

net.load_state_dict(checkpoint['net'])
net.eval()

if isMSE:
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.CrossEntropyLoss()

# t: time information extracted from ckpt file
t = args.file[-13:-4]

RES_DIR = 'result'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

def infer(inputs, targets, num, img_type='um'):
    visualizeCyl(inputs[0].permute(1, 2, 0), file=os.path.join(RES_DIR, '%s_%06d-input-%s.png' % (img_type, num, t)))
    with torch.no_grad():
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        outputs = net(inputs)

        visualizeOutput(inputs, targets, outputs, num, img_type,
            isMSE=isMSE, file=os.path.join(RES_DIR, '%s_%06d-output-%s.png' % (img_type, num, t)))

        if not(isMSE):
            targets = targets.to(torch.long).permute(0, 2, 3, 1).reshape(-1)
            outputs = net(inputs).permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])

        loss = criterion(outputs, targets)
    return loss.item()

types = np.concatenate([[t] * d for d, t in zip(datanum, img_types)])
idx = np.concatenate([np.arange(s, s+d) for d, s in zip(datanum, startidx)])

for i, (inputs, targets) in enumerate(evalloader):
    loss = infer(inputs, targets, idx[i], types[i])
    print('loss of %3s_%06d.png: %.3f' % (types[i], idx[i], loss))
