import libauc
import segmentation_models_pytorch
from libauc.datasets import CheXpert
from libauc.models import DenseNet121
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from libauc.losses import AUCM_MultiLabel, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm




def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# dataloader
# paramaters
SEED = 123
BATCH_SIZE = 16
lr = 1e-4
weight_decay = 1e-5
device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
root = '/mnt/dsi_vol1/shaya/'
# Index: -1 denotes multi-label mode including 5 diseases
traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=False, use_frontal=True, image_size=384, mode='train', class_index=-1)
testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=384, mode='valid', class_index=-1)
trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=BATCH_SIZE, shuffle=True)
testloader =  torch.utils.data.DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)


# model
set_all_seeds(SEED)
model = segmentation_models_pytorch.Unet(
    encoder_name="densenet121",
    encoder_weights="imagenet",
    classes=1,aux_params ={'classes':5}
)

model = model.to(device)
# yy = DenseNet121(num_classes=5)
# define loss & optimizer
CELoss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# training
best_val_auc = 0
for epoch in range(1):
    for idx, data in tqdm(enumerate(trainloader),desc='train',total=len(trainloader)):
        train_data, train_labels = data
        train_data, train_labels  = train_data.to(device), train_labels.to(device)
        y_pred = model(train_data)[1]

        loss = CELoss(y_pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation
        if idx % 400 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = []
                test_true = []
                for jdx, data in tqdm(enumerate(testloader),desc='val'):
                    test_data, test_labels = data
                    test_data = test_data.to(device)
                    y_pred = model(test_data)[1]
                    test_pred.append(y_pred.cpu().detach().numpy())
                    test_true.append(test_labels.numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                val_auc_mean =  roc_auc_score(test_true, test_pred)
                model.train()

                if best_val_auc < val_auc_mean:
                    best_val_auc = val_auc_mean
                    torch.save(model.state_dict(), 'ce_pretrained_model.pth')

                print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc ))