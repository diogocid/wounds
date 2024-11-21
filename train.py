# Code for MedT

import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
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
import timeit
from IPython.display import display

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--num_classes', type=int, required=True, help='number of output classes')

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

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
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.set_deterministic(True)
# random.seed(seed) 
predictions = []
ground_truths = []
predictionsT = []
ground_truthsT = []

train_losses = []
val_losses = []
train_miou_class1 = []
train_miou_class2 = []
val_miou_class1 = []
val_miou_class2 = []

for epoch in range(args.epochs):

    epoch_running_loss = 0
    predictions_train = []
    ground_truths_train = []
    predictionsT = []
    ground_truthsT = []
    
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
        
        X_batch = Variable(X_batch.to(device='cuda'))
        y_batch = Variable(y_batch.to(device='cuda'))
        
        # Forward
        output = model(X_batch)
        
        # Função de perda para multi-classe
        loss = criterion(output, y_batch)
        
        # Backward e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

        predicted_masks = torch.argmax(output, dim=1).detach().cpu().numpy()
        predictions_train.append(predicted_masks)
        ground_truths_train.append(y_batch.detach().cpu().numpy())
        
        predictionsT.append(predicted_masks)
        ground_truthsT.append(y_batch.detach().cpu().numpy())

        
    # Média do loss por batch
    train_loss = epoch_running_loss / (batch_idx + 1)
    train_losses.append(train_loss)

    # Cálculo de métricas de treinamento
    predictions_train = np.array(predictions_train).reshape(-1, *predicted_masks.shape[1:])
    ground_truths_train = np.array(ground_truths_train).reshape(-1, *predicted_masks.shape[1:])
    iou_train, _, _ = calculate_classwise_metrics(predictions_train, ground_truths_train, num_classes=args.num_classes)
    train_miou_class1.append(iou_train[1])
    train_miou_class2.append(iou_train[2])
    
    
    # Converter para numpy arrays
    predictionsT = np.array(predictionsT).reshape(-1, *predicted_masks.shape[1:])
    ground_truthsT = np.array(ground_truthsT).reshape(-1, *predicted_masks.shape[1:])

    # Cálculo de métricas de segmentação
    iou, precision, recall = calculate_classwise_metrics(predictionsT, ground_truthsT, num_classes=args.num_classes)
    
    # Mostrar métricas de segmentação separadas por classe 
    for cls in range(args.num_classes):
        print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")


    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1)))

        # Avaliar métricas
    if (epoch % args.save_freq) == 0:
        predictions = []
        ground_truths = []

        val_running_loss = 0
        predictions_val = []
        ground_truths_val = []
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move inputs e labels para o mesmo dispositivo
            #X_batch = Variable(X_batch.to(device='cuda'))
            y_out = model(X_batch)
            loss = criterion(y_out, y_batch)
            val_running_loss += loss.item()
            
            predicted_masks = torch.argmax(y_out, dim=1).detach().cpu().numpy()
            ground_truths.append(y_batch.detach().cpu().numpy())
            predictions.append(predicted_masks)
            predictions_val.append(predicted_masks)
            ground_truths_val.append(y_batch.detach().cpu().numpy())

            fulldir = os.path.join(direc, f"{epoch}")
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            # Salvar a predição e a máscara ground truth
            cv2.imwrite(os.path.join(fulldir, image_filename), predicted_masks[0])
                    

        # Média do loss de validação
        val_loss = val_running_loss / (batch_idx + 1)
        val_losses.append(val_loss)

        # Cálculo de métricas de validação
        predictions_val = np.array(predictions_val).reshape(-1, *predicted_masks.shape[1:])
        ground_truths_val = np.array(ground_truths_val).reshape(-1, *predicted_masks.shape[1:])
        iou_val, _, _ = calculate_classwise_metrics(predictions_val, ground_truths_val, num_classes=args.num_classes)
        val_miou_class1.append(iou_val[1])
        val_miou_class2.append(iou_val[2])

        print(f"Epoch [{epoch+1}/{args.epochs}], Val Loss: {val_loss:.4f}, Val mIoU Class 1: {iou_val[1]:.4f}, Val mIoU Class 2: {iou_val[2]:.4f}")

        # Converter para numpy arrays
        predictions = np.array(predictions).reshape(-1, *predicted_masks.shape[1:])
        ground_truths = np.array(ground_truths).reshape(-1, *predicted_masks.shape[1:])

        # Cálculo de métricas de segmentação
        iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)
        
        # Mostrar métricas de segmentação separadas por classe 
        for cls in range(args.num_classes):
            print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")


#Salvar o modelo
torch.save(model.state_dict(), os.path.join(fulldir, args.modelname + ".pth"))
torch.save(model.state_dict(), os.path.join(direc, "final_model.pth"))

# Cálculo de métricas de segmentação
iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)

# Mostrar métricas de segmentação separadas por classe 
for cls in range(args.num_classes):
    print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")


# Gráficos
epochs = range(1, args.epochs + 1)

# Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss per Epoch")
plt.savefig("LossperEpoch.png") 
plt.close()

# mIoU Class 1
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_miou_class1, label="Train mIoU Class 1")
plt.plot(epochs, val_miou_class1, label="Validation mIoU Class 1")
plt.xlabel("Epochs")
plt.ylabel("mIoU")
plt.legend()
plt.title("mIoU Class 1 per Epoch") 
plt.savefig("mIoU1_Epocht.png") 
plt.close()

# mIoU Class 2
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_miou_class2, label="Train mIoU Class 2")
plt.plot(epochs, val_miou_class2, label="Validation mIoU Class 2")
plt.xlabel("Epochs")
plt.ylabel("mIoU")
plt.legend()
plt.title("mIoU Class 2 per Epoch")
plt.savefig("mIoU2_Epocht.png") 
plt.close()