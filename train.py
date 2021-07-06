import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torchsummary import summary

import os
import datetime
import argparse

from utility import *
from data_loader import *
from model import ChipNet

import functools
print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser(description='PyTorch KITTI Point Cloud Data Training')
# --mse, -m: use MSE instead of Cross-entropy
parser.add_argument('--mse', '-m', action='store_true', help='use MSE for criterion')
# --progress, -p: use progress bar when preparing dataset
parser.add_argument('--progress', '-p', action='store_true', help='use progress bar')
# --summary, -s: show torchsummary to see the neural net structure
parser.add_argument('--summary', '-s', action='store_true', help='show torchsummary')
args = parser.parse_args()

# img_types = ['um', 'umm', 'uu']: image types in dataset
# total_datanum: number of total images with corresponding img_types (must be list of length 3)
total_datanum = [95, 96, 98]

# train_datanum: number of train images with corresponding img_types (must be list of length 3)
train_datanum = [80] * 3
# test_datanum: number of test images with corresponding img_types (must be list of length 3)
test_datanum = [i - t for i, t in zip(total_datanum, train_datanum)]

n_epoch = 150

print('==> Preparing train data..')
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasetsKitti('data_road', train_datanum, transforms=transform, progress=args.progress)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)
print('==> Preparing test data..')
testset = datasetsKitti('data_road', test_datanum, startidx=train_datanum, transforms=transform, progress=args.progress)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Building model..')
if args.mse:
    net = ChipNet(och=1).to(device)
else:
    net = ChipNet().to(device)
if args.summary:
    summary(net, (14, 180, 64))

if device == 'cuda':
    # Acceleration by using DataParallel
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.mse:
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train():
    net.train()
    train_loss = 0
    for inputs, targets in trainloader:
        optimizer.zero_grad()

        # inputs : [batch_size, channel= 14, height=180, width=64]
        # targets: [batch_size, channel=  1, height=180, width=64]
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        # outputs: [batch_size, channel=och, height=180, width=64]
        outputs = net(inputs)

        if not(args.mse):
            # targets: [batch_size * height * width]
            targets = targets.to(torch.long).permute(0, 2, 3, 1).reshape(-1)
            # outputs: [batch_size * height * width][channel=och=2]
            outputs = net(inputs).permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(trainloader)

def test():
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in trainloader:
            # inputs : [batch_size, channel= 14, height=180, width=64]
            # targets: [batch_size, channel=  1, height=180, width=64]
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            # outputs: [batch_size, channel=och, height=180, width=64]
            outputs = net(inputs)

            if not(args.mse):
                # targets: [batch_size * height * width]
                targets = targets.to(torch.long).permute(0, 2, 3, 1).reshape(-1)
                # outputs: [batch_size * height * width][channel=och=2]
                outputs = net(inputs).permute(0, 2, 3, 1).reshape(-1, outputs.shape[1])

            loss = criterion(outputs, targets)
            test_loss += loss.item()

    return test_loss / len(testloader)

t = datetime.datetime.now().strftime('%m%d-%H%M')
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
CKPT_FILE = './checkpoint/ckpt-%s.pth' % t

epochs = np.arange(1, n_epoch+1)
best_loss = None
train_loss = []
test_loss = []
for epoch in epochs:
    train_loss += [train()]
    test_loss += [test()]
    print('[Epoch: %3d] train loss: %.3f test loss: %.3f' % (epoch, train_loss[-1], test_loss[-1]))

    if (best_loss is None) or (test_loss[-1] < best_loss):
        print('Saving ckeckpoint..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss[-1],
            'epoch': epoch,
            'criterion': 'Cross-entropy' if not(args.mse) else 'MSE'
        }
        torch.save(state, CKPT_FILE)
        best_loss = test_loss[-1]

    scheduler.step()

if not os.path.isdir('graph'):
    os.mkdir('graph')
LOSS_FILE = './graph/loss-%s.png' % t

# visualize loss change
plt.figure()

plt.plot(epochs, train_loss, label="train", color='tab:blue')
am = np.argmin(train_loss)
plt.plot(epochs[am], train_loss[am], color='tab:blue', marker='x')
plt.text(epochs[am], train_loss[am]-0.01, '%.3f' % train_loss[am], horizontalalignment="center", verticalalignment="top")

plt.plot(epochs, test_loss, label="test", color='tab:orange')
am = np.argmin(test_loss)
plt.plot(epochs[am], test_loss[am], color='tab:orange', marker='x')
plt.text(epochs[am], test_loss[am]+0.01, '%.3f' % test_loss[am], horizontalalignment="center", verticalalignment="bottom")

plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.legend()
plt.title('loss')
plt.savefig(LOSS_FILE)