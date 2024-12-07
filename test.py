import argparse
import lib
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D, calculate_classwise_metrics
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint


parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--train_dataset',  type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 5)
parser.add_argument('--modelname', default='off', type=str,
                    help='name of the model to load')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--direc', default='./results', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--num_classes', type=int, required=True, help='number of output classes')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: off)')

args = parser.parse_args()

direc = args.direc
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
loaddirec = args.loaddirec

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
#train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
#dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

model.load_state_dict(torch.load(loaddirec))
model.eval()


# predictions = []
# ground_truths = []
# for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
#     # print(batch_idx)
#     if isinstance(rest[0][0], str):
#                 image_filename = rest[0][0]
#     else:
#                 image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

#     X_batch = Variable(X_batch.to(device='cuda'))
#     y_batch = Variable(y_batch.to(device='cuda'))

#     y_out = model(X_batch)
#     predicted_masks = torch.argmax(y_out, dim=1).detach().cpu().numpy()
#     ground_truths.append(y_batch.detach().cpu().numpy())
#     predictions.append(predicted_masks)
  
# # Converter para numpy arrays
# predictions = np.array(predictions).reshape(-1, *predicted_masks.shape[1:])
# ground_truths = np.array(ground_truths).reshape(-1, *predicted_masks.shape[1:])

# # Cálculo de métricas de segmentação
# iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)
    
# # Mostrar métricas de segmentação separadas por classe 
# for cls in range(args.num_classes):
#     print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")

predictions = []
ground_truths = []
iou_classes = [[] for _ in range(args.num_classes)]
precision_classes = [[] for _ in range(args.num_classes)]
recall_classes = [[] for _ in range(args.num_classes)]

for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    # Identificar o nome do arquivo, se aplicável
    if isinstance(rest[0][0], str):
        image_filename = rest[0][0]
    else:
        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

    X_batch = Variable(X_batch.to(device='cuda'))
    y_batch = Variable(y_batch.to(device='cuda'))

    y_out = model(X_batch)
    predicted_masks = torch.argmax(y_out, dim=1).detach().cpu().numpy()
    yval = y_batch.detach().cpu().numpy()
    ground_truths.append(yval)
    predictions.append(predicted_masks)
    
    fulldir = os.path.join(direc, f"{batch_idx}")
    predicted_masks = predicted_masks * 85  # Escalar para visualização
    # Salvar a predição e a máscara ground truth
    cv2.imwrite(os.path.join(fulldir, image_filename), predicted_masks[0])
    
# Converter para numpy arrays
predictions = np.array(predictions).reshape(-1, *predicted_masks.shape[1:])
ground_truths = np.array(ground_truths).reshape(-1, *predicted_masks.shape[1:])

# Cálculo de métricas de segmentação
iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)

# Mostrar métricas de segmentação separadas por classe 
for cls in range(args.num_classes):
    print(f"Segmentação Test - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")
    iou_classes[cls].append(iou[cls])
    precision_classes[cls].append(precision[cls])
    recall_classes[cls].append(recall[cls])
