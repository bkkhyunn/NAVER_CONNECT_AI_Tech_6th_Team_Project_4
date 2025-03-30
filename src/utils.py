# python native
import os
import random

# external library
from PIL import Image
import cv2
import numpy as np
from skimage.transform import resize
from tqdm.auto import tqdm
import wandb

# torch
import torch

# seed setting
RANDOM_SEED = 666

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

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

# for visualization -> 클래스가 2개 이상인 픽셀을 고려하지는 않음.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

# 겹쳐진 클래스 고려한 시각화
def label2rgba(label, image_np):
    image_size = (512,512) + (4, )  # Add an alpha channel
    result_image = Image.fromarray(np.zeros((512, 512, 4), dtype=np.uint8), 'RGBA')
    result_image.paste(Image.fromarray(image_np, 'RGB'), (0, 0))
    
    for i, class_label in enumerate(label):
        class_label = resize(class_label, (512,512))
        mask = class_label == 1
        image = np.zeros(image_size, dtype=np.uint8)
        image[mask] = PALETTE[i] + (128,)  # Add opacity
        result_image = Image.alpha_composite(result_image, Image.fromarray(image, 'RGBA'))
    
    return result_image


# 실제 정답인데 정답이 아니라고 한 것 (true negative)
def tn2rgb(true, pred, image_np):
    image_size = (512,512) + (4, )  # Add an alpha channel
    result_image = Image.fromarray(np.zeros((512, 512, 4), dtype=np.uint8), 'RGBA')
    result_image.paste(Image.fromarray(image_np, 'RGB'), (0, 0))
    
    for i, pred_label in enumerate(pred):
        pred_label = resize(pred_label, (512,512))
        true_label = resize(true[i], (512,512))
        mistake = pred_label != true_label
        true_negative = mistake & (true_label == 1)
        image = np.zeros(image_size, dtype=np.uint8)
        image[true_negative] = PALETTE[i] + (128,)
        result_image = Image.alpha_composite(result_image, Image.fromarray(image, 'RGBA'))
        
    return result_image
        

# 실제 정답이 아닌데 정답이라고 한 것 (false positive)
def fp2rgb(true, pred, image_np):
    image_size = (512,512) + (4, )  # Add an alpha channel
    result_image = Image.fromarray(np.zeros((512, 512, 4), dtype=np.uint8), 'RGBA')
    result_image.paste(Image.fromarray(image_np, 'RGB'), (0, 0))
    
    for i, pred_label in enumerate(pred):
        pred_label = resize(pred_label, (512,512))
        true_label = resize(true[i], (512,512))
        mistake = pred_label != true_label
        false_positive = mistake & (true_label == 0)
        image = np.zeros(image_size, dtype=np.uint8)
        image[false_positive] = PALETTE[i] + (128,)
        result_image = Image.alpha_composite(result_image, Image.fromarray(image, 'RGBA'))
        
    return result_image

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
