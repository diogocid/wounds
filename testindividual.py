import argparse
import lib
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
from utils import JointTransform2D, ImageToImage2D, Image2D, calculate_classwise_metrics

# Argumentos
parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, help='Number of total epochs to run (default: 1)')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('--train_dataset', type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int, default=5)
parser.add_argument('--modelname', default='off', type=str, help='Name of the model to load')
parser.add_argument('--cuda', default="on", type=str, help='Switch on/off CUDA option (default: off)')
parser.add_argument('--direc', default='./results', type=str, help='Directory to save results')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes')
parser.add_argument('--single_image', type=str, default=None, help='Path to a single image for inference')

args = parser.parse_args()

# Configuração do modelo
imgchant = 1 if args.gray == "yes" else 3
device = torch.device(args.device)
crop = (args.crop, args.crop) if args.crop else None

# Transformações
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)

# Inicialização do modelo
if args.modelname == "axialunet":
    model = lib.models.axialunet(img_size=args.imgsize, imgchan=imgchant)
elif args.modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size=args.imgsize, imgchan=imgchant)
elif args.modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size=args.imgsize, imgchan=imgchant)
elif args.modelname == "logo":
    model = lib.models.axialnet.logo(img_size=args.imgsize, imgchan=imgchant)
else:
    raise ValueError("Invalid model name!")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# Carregar pesos do modelo
model.load_state_dict(torch.load(args.loaddirec))
model.eval()

# Verificar se é para processar uma única imagem
if args.single_image:
    def predict_single_image(image_path, model, device, output_dir, img_size):
        # Carregar e transformar a imagem
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Inferência
        with torch.no_grad():
            output = model(img_tensor)
            predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Salvar predição
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, predicted_mask * 85)
        print(f"Prediction saved to {output_path}")

    output_dir = args.direc
    predict_single_image(args.single_image, model, device, output_dir, args.imgsize)
    exit()

# Caso contrário, processar o conjunto de validação
if not args.val_dataset:
    raise ValueError("--val_dataset is required for validation!")

val_dataset = ImageToImage2D(args.val_dataset, tf_val)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Processar o conjunto de validação
predictions = []
ground_truths = []

for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    # Inferência
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    y_out = model(X_batch)

    # Processar resultados
    predicted_masks = torch.argmax(y_out, dim=1).cpu().numpy()
    ground_truths.append(y_batch.cpu().numpy())
    predictions.append(predicted_masks)

# Calcular métricas
iou, precision, recall = calculate_classwise_metrics(
    np.array(predictions).reshape(-1, *predicted_masks.shape[1:]),
    np.array(ground_truths).reshape(-1, *predicted_masks.shape[1:]),
    num_classes=args.num_classes
)

# Mostrar métricas
for cls in range(args.num_classes):
    print(f"Class {cls} - IoU: {iou[cls]:.4f}, Precision: {precision[cls]:.4f}, Recall: {recall[cls]:.4f}")
