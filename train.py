# -*- coding: utf-8 -*-
import argparse
import os
import random
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import logging
import unet3d
from dataset import ReScale, RandomRotFlip, ToTensor
from tensorboardX import SummaryWriter
from dataset import BraTS2018
from loss import DiceLoss
from torchvision.utils import make_grid
from utils import AverageMeter

patch_size = [96, 128, 128]
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="processed_brats18",
                    help="the root path of Brats18 dataset")
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--batch-size', type=int, default=1, help='random seed')
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="weight decay in optimizer")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--epoch", type=int, default=100, help="train how many epoch")
parser.add_argument("--devices", type=str, default="cpu",
                    help="use which devices to train")
parser.add_argument("--log", type=str, default="log",
                    help="where to place your log files")
parser.add_argument("--checkpoint", type=str, default="checkpoint",
                    help="where to place your weight file")
args = parser.parse_args()

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
filehandler = logging.FileHandler(filename=f"log/{time.time()}.log")
logger.addHandler(filehandler)

writer = SummaryWriter(os.path.join(args.log, "tensorboard"))

db_train = BraTS2018(base_dir=args.data_path,
                     split='train',
                     transform=transforms.Compose([
                         ReScale(patch_size),
                         RandomRotFlip(),

                     ]))

db_test = BraTS2018(base_dir=args.data_path,
                    split='test',
                    transform=transforms.Compose([
                        ReScale(patch_size),
                    ]))
trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                         worker_init_fn=worker_init_fn)
testloader = DataLoader(db_test, batch_size=1, num_workers=4)
network = unet3d.Unet3d(in_dim=2, out_dim=2, num_filter=8)
criterion = DiceLoss()
opt = Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
sched = CosineAnnealingLR(opt, T_max=len(db_train) / args.batch_size * args.epoch, eta_min=1e-7, )
it = 0
test_acc = AverageMeter()
for epoch_num in range(args.epoch):
    tbar = tqdm((trainloader), ncols=70)
    network.train()
    for i_batch, sampled_batch in enumerate(tbar):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.to(args.devices), label_batch.to(args.devices)
        out = torch.softmax(network(volume_batch), dim=1)
        loss = criterion(out, label_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tbar.set_description(f"epoch:{epoch_num} batch idx: {i_batch} dice loss:{loss.cpu().detach().item():.5f}")
        writer.add_scalar("train/loss", loss.cpu().detach().item(), it)
        sample = list(range(0, volume_batch.shape[2], 3))
        sample_img = np.expand_dims(volume_batch[0, 0, sample].cpu().numpy(), axis=1).repeat(3, axis=1)
        pred = np.expand_dims(torch.argmax(out.cpu(), dim=1).int().numpy()[0, sample], axis=1).repeat(3, axis=1)
        mask = np.expand_dims(label_batch[0, sample].cpu().numpy(), axis=1).repeat(3, axis=1)

        grid_img = np.concatenate([sample_img, pred, mask])
        grid_img = make_grid(torch.from_numpy(grid_img), nrow=len(sample))
        writer.add_image("train/img", grid_img, it)
        logger.info(f"TRAIN | epoch:{epoch_num} batch idx: {i_batch} dice loss:{loss.cpu().detach().item():.5f}")
        it += 1

    network.eval()
    with torch.no_grad():
        test_bar = tqdm(testloader, ncols=70)
        for id, test_batch in enumerate(test_bar):
            volume, label = test_batch['image'], test_batch['label']
            volume, label = volume.to(args.devices), label.to(args.devices)
            out = torch.softmax(network(volume), dim=1)
            loss = criterion(out, label)
            test_acc.update(1 - loss.cpu().detach().item())
            test_bar.set_description(
                f"EVAL | epoch:{epoch_num} batch idx: {id} dice loss:{test_acc.value:.5f}")
        logger.info(f"epoch:{epoch_num} test acc: {test_acc.average}")
        if epoch_num % 50 == 0:
            checkpointpth = os.path.join(args.checkpoint, f"epoch{epoch_num}_checkpoint_avg_dice{test_acc.value}.pth")
            torch.save(network.state_dict(), checkpointpth)
            logger.info(f"save checkpoint to {checkpointpth}")
