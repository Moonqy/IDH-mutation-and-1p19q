import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from monai.data import DataLoader, decollate_batch, ImageDataset, CacheDataset, create_test_image_3d, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    RandRotate90d,
    CropForegroundd,
    ScaleIntensityd,
    Activations,
    AsDiscrete,
    EnsureChannelFirstd,
    RandFlipd,
    RandAffined,
    Rand3DElasticd,
    NormalizeIntensityd,
)
import torch
import argparse
from torch.optim import lr_scheduler
from tqdm import tqdm
from monai.metrics import ROCAUCMetric
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, recall_score
from models.IDHNet0 import IDHNet00
from models.IDHNet1 import IDHNet01  # Make sure you have IDHNet2 defined similarly.
from models.IDHNet2 import IDHNet02
from models.IDHNet3 import IDHNet03
from models.IDHNet4 import IDHNet04
from models.IDHNet5 import IDHNet05
from models.IDHNet6 import IDHNet06
from monai.networks.nets import DenseNet121
import torch.nn as nn
import csv
import copy
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--type", default=1, type=int)
parser.add_argument("--model_name", default="IDHNet1", type=str)
parser.add_argument("--loss", default=1e-3, type=float, help="Initial learning rate")  # Added help message
parser.add_argument("--max_epochs", default=80, type=int)
parser.add_argument("--tf", default=0, type=int)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--bn", default=4, type=int, help="Batch size")  # Added help message
parser.add_argument("--file", default=0, type=int)
parser.add_argument("--segmented", default=1, type=int)
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay for AdamW") # Added weight decay
args = parser.parse_args()

pin_memory = False  # Keep as False unless data loading is a bottleneck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.segmented == 1:
    PATH_DATASET = '../../brain-images/segImgs'
    NK = 4
else:
    PATH_DATASET = '/mnt/e/UCSF-PDGM-v3/'
    NK = 4

imgtype = []
if args.type == 0:
    imgtype = ('_T1c_bias', '_DTI_eddy_FA')
elif args.type == 1:
    imgtype = ('_T1c_bias', '_ADC')
elif args.type == 2:
    imgtype = ('_T1_bias', '_DWI_bias')
elif args.type == 3:
    imgtype = ('_T1_bias', '_FLAIR_bias')

auc_metric = ROCAUCMetric()

if args.file==0:
    df_train = pd.read_csv(('../../brain-images/UCSF-IDH-train.csv'),usecols=['ID', 'IDH','age','gender','1p/19q','fold'])
elif args.file==1:
    df_train = pd.read_csv(('../../brain-images/UCSF-IDH-trainm.csv'),usecols=['ID', 'IDH','age','gender','1p/19q','fold'])
images = []

def get_tumor_location(mri_path):
    """
    根据 MRI 文件路径计算肿瘤位置。

    Args:
        mri_path: MRI 文件的路径

    Returns:
        location_code: 肿瘤位置的类别编码 (整数)
    """
    # 1. 加载 MRI
    mri_image = LoadImaged(keys=["img1"])(
        {"img1": mri_path}
    )["img1"]  # 使用 LoadImaged 加载

    # 2. 计算质心
    if isinstance(mri_image, torch.Tensor):
        mri_image = mri_image.cpu().numpy()
    coords = np.array(np.where(mri_image > 0))
    if coords.size == 0:
        return 4  # 如果没有肿瘤区域，返回"其他"类别
    centroid = np.mean(coords, axis=1)

    # 3. 映射到脑区 (简化示例，根据实际情况调整)
    # 假设图像尺寸为 (130, 170, 130)，并且 x, y, z 分别对应左右、前后、上下
    x, y, z = centroid
    if x < 43:
        location_code = 0  # 左侧
    elif x >87:
        location_code = 1  # 右侧
    elif y < 57:
        location_code = 2  # 前部
    elif y > 113:
        location_code = 3 #后部
    else:
        location_code = 4  # 其他

    return location_code


def get_tumor_size(mri_path):
    """
    根据 MRI 文件路径计算肿瘤大小。

    Args:
        mri_path: MRI 文件的路径

    Returns:
        size: 肿瘤大小 (体积)
    """
    # 1. 加载 MRI
    mri_image = LoadImaged(keys=["img1"])(
        {"img1": mri_path}
    )["img1"]  # 使用 LoadImaged 加载

    # 2. 计算体积
    if isinstance(mri_image, torch.Tensor):
        mri_image = mri_image.cpu().numpy()
    volume = np.sum(mri_image > 0)

    return volume

def calculate_average_metrics(csv_file, num_folds=5):
    try:
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"Warning: File '{csv_file}' does not exist.")
            return

        # Read all lines from the file
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        # Check if we have enough lines
        if len(lines) < num_folds:
            print(f"Warning: File has less than {num_folds} lines")
            return

        # Take last num_folds lines
        last_folds = lines[-num_folds:]

        # Initialize sums
        auc_sum = acc_sum = f1_sum = recall_sum = 0

        # Process each line
        for line in last_folds:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "AUC":
                    auc_sum += float(parts[i + 1])
                elif part == "ACC":
                    acc_sum += float(parts[i + 1])
                elif part == "f1":
                    f1_sum += float(parts[i + 1])
                elif part == "recall":
                    recall_sum += float(parts[i + 1])

        # Calculate averages
        auc_avg = auc_sum / num_folds
        acc_avg = acc_sum / num_folds
        f1_avg = f1_sum / num_folds
        recall_avg = recall_sum / num_folds

        # Print results
        print(f"\nAverage metrics for last {num_folds} folds:")
        print(f"Average AUC: {auc_avg:.4f}")
        print(f"Average ACC: {acc_avg:.4f}")
        print(f"Average F1: {f1_avg:.4f}")
        print(f"Average Recall: {recall_avg:.4f}")

        # Append to file
        with open(csv_file, 'a') as f:
            f.write(f"\nAverage metrics for last {num_folds} folds:\n")
            f.write(f"AUC {auc_avg:.4f} ACC {acc_avg:.4f} f1 {f1_avg:.4f} recall {recall_avg:.4f}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

# 全局统计值（直接嵌入）
MEAN_AGE = 56.86666666666667
STD_AGE = 15.054996821418772
MEAN_SIZE = 11.095246161529696  # log(size + 1) 的均值
STD_SIZE = 0.9044013400469199    # log(size + 1) 的标准差

def readIMGs(imgPath, csvFile, imgtype, N):
    train_files = []
    if N == 0:
        tfolds = [1, 2, 3, 4]
    elif N == 1:
        tfolds = [0, 2, 3, 4]
    elif N == 2:
        tfolds = [0, 1, 3, 4]
    elif N == 3:
        tfolds = [0, 1, 2, 4]
    elif N == 4:
        tfolds = [0, 1, 2, 3]
    
    for id, idh, age, gender, fold in zip(csvFile['ID'], csvFile['IDH'], csvFile['age'], csvFile['gender'], csvFile['fold']):
        if int(fold) in tfolds:
            id = id[0:10] + '0' + id[10:]
            img1_path = os.path.join(imgPath, id + imgtype[0] + "_trans.nii.gz")
            img2_path = os.path.join(imgPath, id + imgtype[1] + "_trans.nii.gz")
            
            # 计算位置和大小
            location = get_tumor_location(img1_path)
            size = get_tumor_size(img1_path)
            
            # 预处理
            location_onehot = torch.nn.functional.one_hot(torch.tensor(location), num_classes=5).float()
            gender_onehot = torch.nn.functional.one_hot(torch.tensor(gender), num_classes=2).float()
            size_log = torch.log(torch.tensor(size, dtype=torch.float) + 1)
            size_norm = (size_log - MEAN_SIZE) / (STD_SIZE + 1e-8)
            age_norm = (torch.tensor(age, dtype=torch.float) - MEAN_AGE) / (STD_AGE + 1e-8)
            
            train_files.append({
                "img1": img1_path,
                "img2": img2_path,
                "label": idh,
                "location": location_onehot,
                "size": size_norm,
                "age": age_norm,
                "gender": gender_onehot
            })
    return train_files


def readIMGsTest(imgPath, csvFile, imgtype, foldN):
    test_files = []
    for id, idh, age, gender, fold in zip(csvFile['ID'], csvFile['IDH'], csvFile['age'], csvFile['gender'], csvFile['fold']):
        if int(fold) == foldN:
            id = id[0:10] + '0' + id[10:]
            img1_path = os.path.join(imgPath, id + imgtype[0] + "_trans.nii.gz")
            img2_path = os.path.join(imgPath, id + imgtype[1] + "_trans.nii.gz")
            
            # 计算位置和大小
            location = get_tumor_location(img1_path)
            size = get_tumor_size(img1_path)
            
            # 预处理
            location_onehot = torch.nn.functional.one_hot(torch.tensor(location), num_classes=5).float()
            gender_onehot = torch.nn.functional.one_hot(torch.tensor(gender), num_classes=2).float()
            size_log = torch.log(torch.tensor(size, dtype=torch.float) + 1)
            size_norm = (size_log - MEAN_SIZE) / (STD_SIZE + 1e-8)
            age_norm = (torch.tensor(age, dtype=torch.float) - MEAN_AGE) / (STD_AGE + 1e-8)
            
            test_files.append({
                "img1": img1_path,
                "img2": img2_path,
                "label": idh,
                "location": location_onehot,
                "size": size_norm,
                "age": age_norm,
                "gender": gender_onehot
            })
    return test_files

def transformers():
    if args.tf == 0:
        train_transforms = Compose([
            LoadImaged(keys=["img1", "img2"]),
            EnsureChannelFirstd(keys=["img1", "img2"]),
            ScaleIntensityd(keys=["img1", "img2"]),
            Orientationd(keys=["img1", "img2"], axcodes="RAS"),
            RandRotate90d(keys=["img1", "img2"], prob=0.8, spatial_axes=[0, 2]),
            CropForegroundd(keys=["img1","img2"],source_key="img1",allow_smaller=True),
            Resized(keys=["img1","img2"], spatial_size=(130,170,130)),
        ])
    elif args.tf==1:
        train_transforms = Compose(
        [
            LoadImaged(keys=["img1", "img2"]),
            EnsureChannelFirstd(keys=["img1", "img2"]),
            ScaleIntensityd(keys=["img1", "img2"]),
            RandRotate90d(keys=["img1", "img2"], prob=0.5, spatial_axes=[0, 1]),
            CropForegroundd(keys=["img1","img2"],source_key="img1",allow_smaller=True),
            Rand3DElasticd(keys=["img1","img2"],prob=0.8, sigma_range=(5, 7),magnitude_range=(100, 130),spatial_size=(130,170,130),rotate_range=(np.pi / 36, np.pi, np.pi/ 36),scale_range=(0.15, 0.15, 0.15), padding_mode ='zeros'),
            Resized(keys=["img1","img2"], spatial_size=(130,170,130)),
        ])
    elif args.tf==2:
        train_transforms = Compose(
        [
            LoadImaged(keys=["img1", "img2"]),
            EnsureChannelFirstd(keys=["img1", "img2"]),
            ScaleIntensityd(keys=["img1", "img2"]),
            RandRotate90d(keys=["img1", "img2"], prob=0.5, spatial_axes=[0, 1]),
            CropForegroundd(keys=["img1","img2"],source_key="img1",allow_smaller=True),
            RandAffined(keys=["img1", "img1"],mode=("bilinear", "nearest"),prob=1.0,shear_range=(0.5,0.5,0.5), padding_mode="zeros",),
            Resized(keys=["img1","img2"], spatial_size=(130,170,130)),
        ])
    elif args.tf==3:
        train_transforms = Compose(
        [
            LoadImaged(keys=["img1", "img2"]),
            EnsureChannelFirstd(keys=["img1", "img2"]),
            ScaleIntensityd(keys=["img1", "img2"]),
            RandRotate90d(keys=["img1", "img2"], prob=0.5, spatial_axes=[0, 1]),
            CropForegroundd(keys=["img1","img2"],source_key="img1",allow_smaller=True),
            RandAffined(keys=["img1", "img1"],mode=("bilinear", "nearest"),prob=1.0,shear_range=(0.5,0.5,0.5), padding_mode="zeros",),
            Rand3DElasticd(keys=["img1","img2"],prob=0.8, sigma_range=(5, 7),magnitude_range=(100, 130),spatial_size=(130,170,130),rotate_range=(np.pi / 36, np.pi, np.pi/ 36),scale_range=(0.15, 0.15, 0.15), padding_mode ='zeros'),
            Resized(keys=["img1","img2"], spatial_size=(130,170,130)),
        ])
    val_transforms = Compose(
    [
        LoadImaged(keys=["img1", "img2"]),
        EnsureChannelFirstd(keys=["img1", "img2"]),
        ScaleIntensityd(keys=["img1", "img2"]),
        CropForegroundd(keys=["img1","img2"],source_key="img1",allow_smaller=True),
        Resized(keys=["img1","img2"], spatial_size=(130,170,130)),
    ])
    return train_transforms, val_transforms


# Define transforms
def threshold_at_one(x):
    # threshold at 1
    return x > 0

def calculate_balanced_weights(num_samples_class0, num_samples_class1, smoothing_factor=0.1):
    """
    计算平衡的类别权重
    Args:
        num_samples_class0: 类别0的样本数 (297)
        num_samples_class1: 类别1的样本数 (113)
        smoothing_factor: 平滑因子
    """
    total_samples = num_samples_class0 + num_samples_class1
    
    # 使用sqrt来减少极端类别不平衡的影响
    weight_0 = torch.sqrt(torch.tensor(total_samples / (num_samples_class0 + smoothing_factor)))
    weight_1 = torch.sqrt(torch.tensor(total_samples / (num_samples_class1 + smoothing_factor)))
    
    weights = torch.tensor([weight_0, weight_1])
    
    # 归一化权重到合理范围
    min_weight = weights.min()
    max_weight = weights.max()
    weights = (weights - min_weight) / (max_weight - min_weight + 1e-6)
    weights = weights * 0.9 + 0.1  # 确保最小权重为0.1
    
    return weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1))
        logpt = logpt.squeeze(1)
        pt = torch.exp(logpt)
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            logpt = logpt * alpha_t
        
        loss = -1 * (1 - pt) ** self.gamma * logpt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

if __name__ == "__main__":

    print(f"fold{args.fold}--{args.model_name}--{imgtype}--BN {args.bn}-- {device}")

    trainlist = []
    testlist = []
    trainlist = readIMGs(PATH_DATASET, df_train, imgtype, args.fold)

    train_transforms, val_transforms = transformers()

    train_ds = Dataset(data=trainlist, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.bn, shuffle=True, num_workers=4, pin_memory=False)

    testlist = readIMGsTest(PATH_DATASET, df_train, imgtype, args.fold)
    test_ds = Dataset(data=testlist, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=pin_memory)

    if args.model_name == 'DN121':
        model = DenseNet121(spatial_dims=3, in_channels=2, out_channels=2).to(device)
    elif args.model_name == 'IDHNet0':
        model = IDHNet00().to(device)
    elif args.model_name == 'IDHNet1':
        model = IDHNet01().to(device)
    elif args.model_name == 'IDHNet2':
        model = IDHNet02().to(device)
    elif args.model_name == 'IDHNet3':
        model = IDHNet03().to(device)
    elif args.model_name == 'IDHNet4':
        model = IDHNet04().to(device)
    elif args.model_name == 'IDHNet5':
        model = IDHNet05().to(device)
    elif args.model_name == 'IDHNet6':
        model = IDHNet06().to(device)
    # Add an elif block for IDHNet2 if you have it
    # elif args.model_name == 'IDHNet2':
    #     model = IDHNet2().to(device)
    else:
        raise ValueError(f"Invalid model_name: {args.model_name}")

    # --- Calculate class weights DYNAMICALLY ---
    train_labels = [data['label'] for data in trainlist]  # Extract labels from trainlist
    num_class_0 = train_labels.count(0)
    num_class_1 = train_labels.count(1)
    total_samples = len(train_labels)

    # 替换原来的 class_weights 计算
    class_weights = calculate_balanced_weights(num_class_0, num_class_1)  # 根据您的数据分布
    class_weights = class_weights.to(device)
    '''
    class_weights = [total_samples / num_class_0, total_samples / num_class_1]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)'
    '''
    #loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_function = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.loss, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, eps=1e-6)  # Reduce LR on AUC plateau
    max_epochs = args.max_epochs 
    total_steps = len(train_loader) * args.max_epochs 
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,  # 最大学习率
        total_steps=total_steps,
        pct_start=0.3,  # 预热阶段占总步数的比例
        div_factor=25,  # 初始学习率 = max_lr/div_factor
        final_div_factor=1e4  # 最终学习率 = max_lr/final_div_factor
    )
    # Use the command line argument
    val_interval = 2
    best_roc_auc = -1
    best_acc = -1
    best_AUC_epoch = -1
    earlyend = 6
    epoch_loss_values = []
    bestauc = -1
    minloss = 100
    losscount = 0
    best_cm = None
    best_roc_data = {
        "fpr": None,
        "tpr": None,
        "auc": None,
        "acc": None,
        "epoch": None
    }

    for epoch in range(max_epochs):
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        epoch_loss = 0
        model.train()
        epoch_iterator_train = tqdm(train_loader)
        for step, batch_data in enumerate(epoch_iterator_train):
            step = step + 1
            inputs1,inputs2,age,gender,location, size, label = batch_data['img1'].to(device),batch_data['img2'].to(device), batch_data['age'].to(device),batch_data['gender'].to(device),batch_data['location'].to(device),batch_data['size'].to(device),batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2,location, size,age,gender)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        # No scheduler.step(loss) here. We'll step based on validation AUC.

        epoch_loss /= step

        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in test_loader:
                valimg1,valimg2,age,gender,location, size, val_labels = val_data["img1"].to(device), val_data["img2"].to(device), val_data['age'].to(device),val_data['gender'].to(device),val_data['location'].to(device),val_data['size'].to(device),val_data["label"].to(device)
                outputs=model(valimg1, valimg2,location, size,age,gender)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)

            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

            f1 = f1_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')
            recall = recall_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')

            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()

            epoch_loss_values.append(epoch_loss)
            print(
                f"-{args.model_name}-{args.type}-fold{args.fold}-BN {args.bn}-epoch {epoch + 1} average loss: {epoch_loss:.4f} AUC: {auc_result:.4f} acc {acc_metric:.4f}  f1 {f1:.4f} recall {recall:.4f}")

            if auc_result > bestauc + 0.001:
                losscount = 0
                bestauc = auc_result
                bestacc = acc_metric
                bestf1 = f1
                bestrecall = recall
                best_epoch = epoch + 1 # Keep track of the best epoch.
                if auc_result>0.96:
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())  # Save optimizer state too!
                    save_path = f"tempModel/{args.type}_{args.model_name}_fold{args.fold}_auc_{bestauc:.4f}_epoch_{best_epoch}.pth" # save path
                    torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': best_state_dict,
                        'optimizer_state_dict': best_optimizer_state,
                        'loss': epoch_loss,  # You can save other things here too.
                        'auc': bestauc,
                        'acc': bestacc,
                        'f1': bestf1,
                        'recall': bestrecall
                    }, save_path)

                    print(
                        f"saved best model to {save_path},best AUC:{auc_result:.4f}  acc {acc_metric:.4f}  f1 {f1:.4f} recall {bestrecall:.4f}")
            else:
                losscount = losscount + 1
    

            if losscount >= earlyend and epoch > 20:
                print('early end!')
                break

    print(
        f"fold {args.fold} {args.model_name} type {imgtype} file {args.file}",
        f" AUC {bestauc:.4f} ACC {bestacc:4f} f1 {bestf1:.4f} recall {bestrecall:.4f}\n")

    result_data = [f"fold {args.fold} {args.model_name} type {imgtype} file {args.file} lr {args.loss} wd {args.weight_decay}",
                   f" AUC {bestauc:.4f} ACC {bestacc:4f} f1 {bestf1:.4f} recall {bestrecall:.4f}"]

    if args.fold == 4:
        pd.DataFrame([result_data]).to_csv(os.path.join('/home/qy/pycode/MGMT/csv/IDH.csv'), mode='a', header=False,
                                           index=False)  # mode??a,????csv???????
        calculate_average_metrics('/home/qy/pycode/MGMT/csv/IDH.csv')
        pd.DataFrame([[]]).to_csv(os.path.join('/home/qy/pycode/MGMT/csv/IDH.csv'), mode='a', header=False, index=False)
    else:
        pd.DataFrame([result_data]).to_csv(os.path.join('/home/qy/pycode/MGMT/csv/IDH.csv'), mode='a', header=False,
                                           index=False)  # mode??a,????csv???????

    torch.cuda.empty_cache()