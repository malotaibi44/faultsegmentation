from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc  # Dice Coefficient from MedPy
import pandas as pd 
import torch 

from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from val import evaluate
from cfg import _c as cfgg
from monai.losses import DiceCELoss
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset
from medpy.metric.binary import dc  # Dice Coefficient from MedPy
from torch.nn import functional as F
from torchvision import transforms
import os 
import numpy as np 
from PIL import Image
import cv2 
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import torch
from torch.utils.data import DataLoader
# from transformers import AutoImageProcessor, UNetForImageSegmentation
# from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm



fp='/home/hice1/malotaibi44/scratch/segformer'
class f3Sec(Dataset):
    label_rgb_codes = {'certain': [31, 119, 180], 'uncertain': [44, 160, 44], 'no': [255, 127, 14]}

    def __init__(self, cfg, datalist, subset,classlabel):
        self.train=[]
        self.lbls=[]
        self.labelset = cfg.dataset.labelset.split('-')
        #groups=os.listdir(cfg.dataset.root)
        grps= [classlabel]
        if subset == 'train': # for train set 
            for group in grps: # take all annotators 
                if not group =='expert':
                    for i,img in enumerate(datalist):
                        
                        #print(os.path.join('images/',img))
                        if os.path.exists(os.path.join('Fault segmentations',group,img)) and os.path.exists(os.path.join('images/',img)):
                            self.train.append(os.path.join('NeurIPS2024_SegFaults-main/images/images/',img))
                            self.lbls.append(os.path.join('Fault segmentations',group,img))
                else:
                    lst=os.listdir(os.path.join(cfg.dataset.root,group))
                    for img in lst:
                        self.train.append(os.path.join('segmented_images',img))
                        self.lbls.append(os.path.join(cfg.dataset.root,group,img))
        else: # for test set 
            for group in grps:
                if any(label in group for label in classlabel) : # check if the annotator in the test class 
                    if not group =='expert':
                        for i,img in enumerate(datalist):
                            
                            if os.path.exists(os.path.join(cfg.dataset.root,group,img)):
                                self.lbls.append(os.path.join(cfg.dataset.root,group,img))
                                self.train.append(os.path.join('NeurIPS2024_SegFaults-main/images/images/',img))
                    else:
                        train=os.listdir(os.path.join('Fault segmentations',group))
                        test=os.listdir(os.path.join('expert'))
                        print(os.path.join('expert'))
                        for img in test:
                            if not img in train:
                                if os.path.exists(os.path.join('NeurIPS2024_SegFaults-main/images/images/',img)):
                                    self.train.append(os.path.join('NeurIPS2024_SegFaults-main/images/images/',img))
                                    self.lbls.append(os.path.join('expert',img))
        #print("fn",len(self.lbls))
    def _get_singleannot_label(self, labels_path):
        label_rgb_array = np.asarray(Image.open(labels_path).convert('RGB'))
        
        ## collect all types of labels
        lblmask = 0.
        for lbl in self.labelset:
            # get the label mask
            _lblmask = get_lblmask(label_rgb_array, np.array(self.label_rgb_codes[lbl]))
            # print(_lblmask.shape, _lblmask.max(), _lblmask.sum())
            lblmask += _lblmask

        label = (lblmask == 255).astype(np.uint8)
        

        return label  
        
    def __getitem__(self, index):
        # Load the image and convert it to RGB
        sec = np.asarray(Image.open(self.train[index]).convert("RGB"))

        # Load the label
        lbl = self._get_singleannot_label(self.lbls[index])

        # Define transformations
        pad_transform = transforms.Pad((0, 0, 0, 701 - 255), fill=0)  # Pad to 701x701
        resize_transform = transforms.Resize((672, 672))  # Resize to 512x512
        resize = transforms.Resize((672, 672))  # Resize to 512x512
        to_tensor_transform = transforms.ToTensor()

        # Apply padding and resizing to the image
        sec_pil = Image.fromarray(sec)
        sec_pil = pad_transform(sec_pil)
        sec_pil = resize_transform(sec_pil)
        #sec_pil = to_tensor_transform(sec_pil)

        # Apply padding and resizing to the label
        lbl_pil = Image.fromarray(lbl)
        lbl_pil = pad_transform(lbl_pil)
        lbl_pil = resize(lbl_pil)

        #print(lbl_pil.shape)
        #lbl_pil = F.one_hot( transforms.ToTensor()(lbl_pil), num_classes=2)
        #lbl_pil = to_tensor_transform(lbl_pil).long()  # Ensure the label is a long tensor
        return torch.tensor(minmaxScalar(np.transpose(sec_pil,(2,0,1))),dtype=torch.float32), torch.tensor(np.asarray(lbl_pil), dtype=torch.long)
    def __len__(self):
        return len(self.train)
def make_dataset(cfg,subset, classannot):
    # get the train/test lists.
    
    trainlist_path = os.path.join('train.txt')
    testlist_path = os.path.join('test.txt')

    with open(trainlist_path, "r") as f:
        trainlist = f.read().split('\n')[:-1]
    #trainlist = [int(l) for l in trainlist]

    with open(testlist_path, "r") as f:
        testlist = f.read().split('\n')[:-1]
    #testlist = [int(l) for l in testlist]
    if subset=='train':
        dataset = f3Sec(cfg, trainlist, 'train',classannot)
    else:
        dataset = f3Sec(cfg, testlist, 'test',classannot) 
    # img,lbl=dataset.__getitem__(0)
    return dataset
def make_dataloader(cfg, subset, classannot):
    # create all dataloaders
    dataset = make_dataset(cfg,subset, classannot)
    if subset=='test':
        loader = DataLoader(dataset = dataset,
                                  batch_size = 32,
                                  num_workers = cfg.dataloader.num_workers,
                                  shuffle = False
                                 )
    else:
        loader = DataLoader(dataset = dataset,
                          batch_size = 32,
                          num_workers = cfg.dataloader.num_workers,
                          shuffle = True
                         )
    return loader,dataset
def minmaxScalar(img):
    return (img - img.min()) / (img.max() - img.min()).astype(np.float32)


def get_lblmask(img, th):
    return cv2.inRange(img, th, th)


criterion = nn.CrossEntropyLoss()
results_dict = {
    'Model': [],
   'mIoU': [],
 'F1 Score': [],
 'Hausdorff Distance': [],
'Dice Score': []
}
for i in range(1,28):
    cat='novice'+"{:02}".format(i)
    trainloader,trainset = make_dataloader(cfgg,'train', cat)
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Define the model
    model = smp.UnetPlusPlus(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )
    #model =torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model.train()

# Define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in tqdm(trainloader):
            pixel_values,labels = batch
            masks  =labels.to(device)
            pixel_values = pixel_values.to(device)
            outputs = model(pixel_values)
            
            loss = loss = criterion(outputs, masks)
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}")

    print('Training complete.')
    torch.save(model,f'unet++_models_inter/{cat}.pth')
   # expert_loader,expert_data=make_dataloader(cfgg,'test','expert')
   # expert_results = evaluate_model(cat,model, expert_loader, device)
   # results_dict['Model'].append(cat)
#    results_dict['mIoU'].append(expert_results['mIoU'])
 #   results_dict['F1 Score'].append(expert_results['F1 Score'])
  #  results_dict['Hausdorff Distance'].append(expert_results['Hausdorff Distance'])
   # results_dict['Dice Score'].append(expert_results['Dice Score'])
    # Convert the results dictionary to a DataFrame
#    print("expert Results:", expert_results)  
 #   results_df = pd.DataFrame(results_dict)
#output_file = 'suplmentary/s_unet_model_evaluation_results_1.xlsx'
#results_df.to_excel(output_file, index=False)
