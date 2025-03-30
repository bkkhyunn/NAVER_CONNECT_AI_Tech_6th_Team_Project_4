# python native
import os
import datetime

# external library
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
import cv2
from PIL import Image
import argparse

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler,RandomSampler,SequentialSampler
from torchvision import models
from torchvision.transforms.functional import to_pil_image
import segmentation_models_pytorch as smp

from dataset import *
from utils import set_seed, label2rgba, fp2rgb, tn2rgb, dice_coef
from loss import BCEDiceLoss, CELoss
from model import *
import wandb

import warnings

warnings.filterwarnings("ignore")

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# 모델 저장
def save_model(model, save_dir, file_name='best.pt'):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)

# loss 튀는 구간 체크
def loss_check(epoch, loss, pre_loss, file_list, save_dir, serial):
    os.makedirs(os.path.join(save_dir, serial), exist_ok=True)
    path = os.path.join(save_dir,serial,'loss_check.txt')
    image_paths = '\n'.join([p for p in file_list])
    with open(path, 'a', encoding='utf-8') as file:
        file.write("="*10+f" train_{epoch+1} "+"="*10 +"\n")
        file.write(f"loss : {loss}" + "\n")
        file.write(f"pre_loss : {pre_loss}" + "\n")
        file.write(f"increase : {loss - pre_loss}" + "\n")
        file.write(image_paths + "\n")
        file.write("\n")

# wandb 시각화
def visualize_and_log_wandb(results, epoch, gray=False):
    for result in tqdm(results, total=len(results)):
        for output, mask, image, image_path in result:

            output_np = output.numpy()
            mask_np = mask.numpy()
            image_np = np.array(image)
            image_np = (image_np.transpose(1, 2, 0) * 255.).astype('uint8')
            if gray:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            image_np = np.array(Image.fromarray(image_np).resize((512, 512)))
            file_name = '/'.join(image_path.split('/')[-2:])

            output_rgba = label2rgba(output_np, image_np)
            mask_rgba = label2rgba(mask_np, image_np)
            fp_rgba = fp2rgb(mask_np, output_np, image_np)
            tn_rgba = tn2rgb(mask_np, output_np, image_np)
            gt_img = image_np

            #Log images to wandb
            wandb.log({f"images on {epoch} epoch validation": 
                    [wandb.Image(gt_img, caption=f"GT Image \n {file_name}"),
                    wandb.Image(mask_rgba, caption="GT Mask"),
                    wandb.Image(output_rgba, caption="Model Prediction"),
                    wandb.Image(fp_rgba, caption="false positive"),
                    wandb.Image(tn_rgba, caption="true negative")]
                    })
    
    
# validation
def validation(epoch, model, val_loader, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    results = []
    matchs = []
    with torch.no_grad():

        for step, (images, masks, image_paths, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()    
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            #pred = torch.argmax(label, dim=-1)
            #acc_item = (aux_label == pred).sum().item()
            #matchs.append(acc_item)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            
            dice = dice_coef(masks, outputs)
            dices.append(dice)
            
            # 시각화용 result
            if (step + 1) % 20 == 0:
                result = [(o, m, i, p) for o, m, i, p in zip(outputs.detach().cpu(), masks.detach().cpu(), images.detach().cpu(), image_paths)]
                results.append(result)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    #acc = np.sum(matchs) / len(val_loader)
    dice_str = "\n".join(dice_str)
    print(dice_str)
    #print('\n'+f'auxiliary acc : {acc}')
    
    avg_dice = torch.mean(dices_per_class).item()

    # Log dice scores per class
    wandb.log({f"Dice Score/{c}": d.item() for c, d in zip(CLASSES, dices_per_class)})
    wandb.log({"Average Dice Score": avg_dice})
    
    return avg_dice, dice_str, results


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    serial = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #aux_criterion = CELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in range(args.num_epochs):
        model.train()
        #matchs = 0

        pre_loss = 100.
        for step, (images, masks, image_paths, _) in enumerate(train_loader):            
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()        
            #outputs = model(images)['out']
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)
            #aux_loss = aux_criterion(label, aux_label)
            
            #total_loss = loss + (aux_loss * 0.1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            #loss.backward()
            scaler.step(optimizer)
            #optimizer.step()
            scaler.update()
            
            #pred = torch.argmax(label, dim=-1)
            #matchs += (aux_label == pred).sum().item()
            
            if (step + 1) % 20 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.num_epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}, '
                    #f'acc: {matchs / (args.batch_size) / 20}, '
                    #f'total_Loss: {round(total_loss.item(),4)}'
                )
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({'Loss': round(loss.item(),4), 'learning rate':current_lr})
                
                if round(loss.item(), 4) > pre_loss and (round(loss.item(), 4) - pre_loss > 0.1):
                    loss_check(epoch+1, round(loss.item(), 4), pre_loss, image_paths, args.save_dir, serial)
                    pre_loss = round(loss.item(), 4)
                else:
                    pre_loss = round(loss.item(), 4)
                    
                matchs = 0
             
        scheduler.step()
        
        # validation
        if (epoch + 1) % args.val_interval == 0:
            dice, dice_log, results = validation(epoch+1, model, val_loader)
            
            wandb.log({'dice score': dice})
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {os.path.join(args.save_dir, serial)}")
                best_dice = dice
                save_model(model, os.path.join(args.save_dir, serial), file_name=f'{epoch+1}.pt')
                save_model(model, os.path.join(args.save_dir, serial), file_name='best.pt')
                
                dice_per_class_log = os.path.join(args.save_dir, serial, 'dice_per_class.txt')
                with open(dice_per_class_log, 'a', encoding='utf-8') as file:
                    file.write("="*10+f" validation_{epoch+1} "+"="*10 +"\n")
                    file.write(dice_log)
                    file.write("\n")
            
            print("\n" + "Wandb image logging ...")
            start_time = datetime.datetime.now()
            visualize_and_log_wandb(results, epoch+1, args.gray)
            end_time = datetime.datetime.now()
            
            elapsed_time = end_time - start_time
            print("\n" + f"==>> 총 visualization 시간: {str(elapsed_time)}")
                
                
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--image_root', help='train image file path', default='/data/ephemeral/home/data/train/DCM')
    parser.add_argument('--label_root', help='label image file path', default='/data/ephemeral/home/data/train/outputs_json')
    parser.add_argument('--save_dir', help='save pt dir path', default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06/pts')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--ms_train', type=bool, default=False)
    args = parser.parse_args()

    return args


def main(args):
    
    # seed 666
    set_seed()
    
    # transform 정의
    tf = A.Resize(2048, 2048)
    #tf = A.Compose([A.GridDistortion(p=0.3)])
    
    if args.ms_train:
        train_dataset = XRayMultiScaleDataset(args.image_root, args.label_root, is_train=True, transforms=None, gray=args.gray)
        valid_dataset = XRayMultiScaleDataset(args.image_root, args.label_root, is_train=False, transforms=None, gray=args.gray)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=BatchSampler(SequentialSampler(train_dataset),
                                    batch_size=args.batch_size,
                                    multiscale_step=1,
                                    drop_last=True,
                                    img_sizes=[512, 768, 1024]),
            num_workers=args.batch_size,
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
        
    else:
        train_dataset = XRayDataset(args.image_root, args.label_root, is_train=True, transforms=tf, gray=args.gray)
        valid_dataset = XRayDataset(args.image_root, args.label_root, is_train=False,  transforms=tf, gray=args.gray)

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.batch_size,
            drop_last=True,
        )
        
        valid_loader = DataLoader(
            dataset=valid_dataset, 
            batch_size=2,
            shuffle=False,
            num_workers=2,
            drop_last=False
        )
    
    #model = models.segmentation.fcn_resnet101(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    # model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    # model.classifier[-1] = torch.nn.Conv2d(256, len(CLASSES), kernel_size=(1, 1))
    aux_params=dict(
        pooling='max',            
        dropout=0.5,              
        activation='softmax',     
        classes=3,                
    )
    
    model = smp.DeepLabV3(
        encoder_name='resnet101',
        encoder_weights="imagenet",
        in_channels=1,
        classes=len(CLASSES),
        aux_params=aux_params
    )
    model = HRUNet3Plus(n_channels=1)
    
    if args.gray:
        pass
    
    criterion = BCEDiceLoss(bce_weight=1)
    #optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    #optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_epochs//2], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs//4, eta_min=1e-6)
    
    train(args, model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == '__main__':
    args = parse_args()
    
    wandb.init(project="CV06_Segmantation",
               #entity="innovation-vision-tech",
               name=f"smp_UNet3+_HRNet_w32_{args.num_epochs}e_BCEDicecloss(dice3,bce1)_AdamW_CosineAnealingLR_shuffle_false_ms_train",
               notes="",
               config={
                    "batch_size": args.batch_size,
                    "Learning_rate": args.lr,
                    "Epochs": args.num_epochs,
                })
    
    start_time = datetime.datetime.now()
    main(args)
    end_time = datetime.datetime.now()
    
    elapsed_time = end_time - start_time
    print("\n" + f"==>> 총 학습 시간: {str(elapsed_time)}" + "\n")
