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
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit

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


# Função para calcular mIoU, Precision e Recall.
def calculate_classwise_metrics(preds, labels, num_classes):
    """
    Calcula métricas separadas por classe: mIoU, Precision e Recall.
    """
    iou_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    count_per_class = np.zeros(num_classes)

    for pred, label in zip(preds, labels):
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            label_cls = (label == cls)

            # # Interseção e união para IoU
            intersection = np.logical_and(pred_cls, label_cls).sum()
            union = np.logical_or(pred_cls, label_cls).sum()
            if union > 0:
                iou_per_class[cls] += intersection / union
                count_per_class[cls] += 1

            # Cálculo da Precision e Recall
            tp = intersection
            fp = pred_cls.sum() - tp
            fn = label_cls.sum() - tp

            if tp + fp > 0:
                precision_per_class[cls] += tp / (tp + fp)
            if tp + fn > 0:
                recall_per_class[cls] += tp / (tp + fn)

    # Média por classe
    iou_per_class = iou_per_class / np.maximum(count_per_class, 1)
    precision_per_class = precision_per_class / np.maximum(count_per_class, 1)
    recall_per_class = recall_per_class / np.maximum(count_per_class, 1)

    return iou_per_class, precision_per_class, recall_per_class

def calculate_metrics(predictions, ground_truths, num_classes):
    """
    Calcula métricas (IoU, Precision e Recall) por classe e mIoU geral.
    """
    #iou_per_class = np.zeros(num_classes)
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    tp = np.zeros(num_classes)  # Verdadeiros Positivos
    fp = np.zeros(num_classes)  # Falsos Positivos
    fn = np.zeros(num_classes)  # Falsos Negativos

    for pred, gt in zip(predictions, ground_truths):
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
            
            tp[cls] += np.logical_and(pred_cls, gt_cls).sum()
            fp[cls] += np.logical_and(pred_cls, ~gt_cls).sum()
            fn[cls] += np.logical_and(~pred_cls, gt_cls).sum()

    # Calcular métricas por classe
    for cls in range(num_classes):
        # union = tp[cls] + fp[cls] + fn[cls]
        # iou_per_class[cls] = tp[cls] / union if union > 0 else 0
        precision_per_class[cls] = tp[cls] / (tp[cls] + fp[cls]) if tp[cls] + fp[cls] > 0 else 0
        recall_per_class[cls] = tp[cls] / (tp[cls] + fn[cls]) if tp[cls] + fn[cls] > 0 else 0

    #miou = iou_per_class.mean()  # Média dos IoUs

    return precision_per_class, recall_per_class# ,iou_per_class, miou


predictions = []
ground_truths = []

for epoch in range(args.epochs):

    epoch_running_loss = 0
    
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
        
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1)))

    # # Salvar modelo e previsões em frequências definidas
    # if (epoch % args.save_freq) == 0:
    #     predictions = []
    #     ground_truths = []
    #     for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    #         if isinstance(rest[0][0], str):
    #             image_filename = rest[0][0]
    #         else:
    #             image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

    #         X_batch = Variable(X_batch.to(device='cuda'))
    #         y_out = model(X_batch) 
            
    #         tmp = y_out.detach().cpu().numpy()
            
    #         yHaT = tmp
            
    #         #yHaT[yHaT==1] =255
    #         # print("yHaT:",yHaT)

    #         # print("y_out:",y_out)
    #         # print("y_out-shape:",y_out.shape)
    #         #y_out = matriz com 256,256,3
    #         # Processamento para multi-classe
    #         predicted_masks = torch.argmax(y_out, dim=1).detach().cpu().numpy() 
    #         #predicted_masks = matriz com 0,1,2
    #         # print("predicted_masks:",predicted_masks)
    #         # print("predicted_masks-shape:",predicted_masks.shape)
    #         predicted_masks = predicted_masks * 85  # Escalar para visualização
    #         yval = y_batch.detach().cpu().numpy() * 85
    #         # print("predicted_masks85:",predicted_masks)
    #         # print("predicted_masks85-shape:",predicted_masks.shape)
    #         # print("yval:",yval)
    #         # print("yval-shape:",yval.shape)

    #         #predicted_masks = matriz com 0,1,2
    #         predictions.append(predicted_masks[0])
    #         ground_truths.append(yval)
            
    #         fulldir = os.path.join(direc, f"{epoch}")
    #         if not os.path.isdir(fulldir):
    #             os.makedirs(fulldir)

    #         # Salvar a predição e a máscara ground truth
    #         cv2.imwrite(os.path.join(fulldir, image_filename), predicted_masks[0])
                    
    #     # Salvar o modelo
    #     torch.save(model.state_dict(), os.path.join(fulldir, args.modelname + ".pth"))
    #     torch.save(model.state_dict(), os.path.join(direc, "final_model.pth"))
        # Avaliar métricas
    if (epoch % args.save_freq) == 0:
        predictions = []
        ground_truths = []

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            X_batch = Variable(X_batch.to(device='cuda'))
            y_out = model(X_batch)
            
            predicted_masks = torch.argmax(y_out, dim=1).detach().cpu().numpy()
            ground_truths.append(y_batch.detach().cpu().numpy())
            predictions.append(predicted_masks)

        # Converter para numpy arrays
        predictions = np.array(predictions).reshape(-1, *predicted_masks.shape[1:])
        ground_truths = np.array(ground_truths).reshape(-1, *predicted_masks.shape[1:])

        # Cálculo de métricas de segmentação
        iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)
        
        # Mostrar métricas de segmentação separadas por classe 
        for cls in range(args.num_classes):
            print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")

        # mIoU Geral (média das classes)
        miou = np.nanmean(iou)
        print(f"Epoch {epoch}/{args.epochs} - mIoU: {miou:.4f}")

        # No loop de validação
        precision_per_class, recall_per_class = calculate_metrics(predictions, ground_truths, args.num_classes)

        print(f"Epoch {epoch}/{args.epochs} - mIoU: {miou:.4f}")
        for cls in range(args.num_classes):
            print(f"Detecção -Class {cls} - Precision: {precision_per_class[cls]:.4f}, Recall: {recall_per_class[cls]:.4f}")



# Cálculo de métricas de segmentação
iou, precision, recall = calculate_classwise_metrics(predictions, ground_truths, num_classes=args.num_classes)

# Mostrar métricas de segmentação separadas por classe 
for cls in range(args.num_classes):
    print(f"Segmentação - Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")

print(f"Epoch {epoch}/{args.epochs} - mIoU: {miou:.4f}")

# No loop de validação
precision_per_class, recall_per_class = calculate_metrics(predictions, ground_truths, args.num_classes)

for cls in range(args.num_classes):
    print(f"Detecção - Class {cls} - Precision: {precision_per_class[cls]:.4f}, Recall: {recall_per_class[cls]:.4f}")



