from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc  # Dice Coefficient from MedPy
import pandas as pd 
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerConfig
import torch.optim as optim
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import cv2
import os
from cfg import _c as cfg
#from f3sec import F3Sec

from torch.nn import functional as F



def minmaxScalar(img):
    return (img - img.min()) / (img.max() - img.min()).astype(np.float32)


def get_lblmask(img, th):
    return cv2.inRange(img, th, th)

#better to create your own. mine is 
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
                        
                        #print(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img))
                        if os.path.exists(os.path.join(fp,'Fault segmentations',group,img)) and os.path.exists(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img)):
                            self.train.append(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img))
                            self.lbls.append(os.path.join(fp,'Fault segmentations',group,img))
                else:
                    lst=os.listdir(os.path.join(fp,cfg.dataset.root,group))
                    for img in lst:
                        self.train.append(os.path.join(fp,'segmented_images',img))
                        self.lbls.append(os.path.join(fp,cfg.dataset.root,group,img))
        else: # for test set 
            for group in grps:
                if any(label in group for label in classlabel) : # check if the annotator in the test class 
                    if not group =='expert':
                        for i,img in enumerate(datalist):
                            
                            if os.path.exists(os.path.join(fp,cfg.dataset.root,group,img)):
                                self.lbls.append(os.path.join(fp,cfg.dataset.root,group,img))
                                self.train.append(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img))
                    else:
                        train=os.listdir(os.path.join(fp,'Fault segmentations',group))
                        test=os.listdir(os.path.join(fp,'expert'))
                        print(os.path.join(fp,'expert'))
                        for img in test:
                            if not img in train:
                                if os.path.exists(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img)):
                                    self.train.append(os.path.join(fp,'NeurIPS2024_SegFaults-main/images/images/',img))
                                    self.lbls.append(os.path.join(fp,'expert',img))
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
        resize_transform = transforms.Resize((512, 512))  # Resize to 512x512
        resize = transforms.Resize((128, 128))  # Resize to 512x512
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, SegformerConfig
from torchvision import transforms
import os
from PIL import Image
#from f3sec import F3Sec
from PIL import Image
from tqdm import tqdm
fp='/home/hice1/malotaibi44/scratch/segformer'

# Define a simple function to save masks
def make_dataset(cfg,subset, classannot):
    # get the train/test lists.
    
    trainlist_path = os.path.join(fp,'train.txt')
    testlist_path = os.path.join(fp,'test.txt')
    with open(trainlist_path, "r") as f:
        trainlist = f.read().split('\n')[:-1]
    #trainlist = [int(l) for l in trainlist]
    #print(trainlist_path)
    with open(testlist_path, "r") as f:
        testlist = f.read().split('\n')[:-1]
    #testlist = [int(l) for l in testlist]
    if subset=='train':
        dataset = f3Sec(cfg, trainlist, 'train',classannot)
    else:
        dataset = f3Sec(cfg, testlist, 'test',classannot) 
    # img,lbl=dataset.__getitem__(0)
    print("n",len(trainlist))
    return dataset
def make_dataloader(cfg, subset, classannot):
    # create all dataloaders
    dataset = make_dataset(cfg,subset, classannot)
    if subset=='test':
        print("Here")
        loader = DataLoader(dataset = dataset,
                                  batch_size = cfg.solver.batch_size,
                                  num_workers = cfg.dataloader.num_workers,
                                  shuffle = False
                                 )
    else:
        loader = DataLoader(dataset = dataset,
                          batch_size = cfg.solver.batch_size,
                          num_workers = cfg.dataloader.num_workers,
                          shuffle = True
                         )


    return loader,dataset
#train_loader = make_dataloader(cfg,'train','expert')
def save_image(tensor, path):
    image = Image.fromarray(tensor.byte().cpu().numpy())
    image.save(path)

# Define the Segformer model with random weights
class SegformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SegformerModel, self).__init__()
        # Load pretrained model
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=num_classes,
                                                                      ignore_mismatched_sizes=True)
        self.model.decode_head.classifier = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    def forward(self, x):
        return self.model(x).logits
    
    def forward(self, x):
        return self.model(x).logits

# Initialize feature extractor
feature_extractor = SegformerFeatureExtractor(size=512)

# Hyperparameters
num_epochs = 100
learning_rate = 1e-3
save_interval = 10
num_classes = 2

results_dict = {
    'Model': [],
   'mIoU': [],
 'F1 Score': [],
 'Hausdorff Distance': [],
'Dice Score': []
}
num_epochs = 100
learning_rate = 1e-3
save_interval = 10
num_classes = 2
for i in range(1,28):
    cat='novice'+"{:02}".format(i)
    # Initialize the model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =SegformerModel(num_classes=2).to(device)
    #model = nn.DataParallel(model)  # Wrap the model for DataParallel SegformerModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print(cat)
    train_loader, dataset = make_dataloader(cfg,'train',cat)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
     
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            #print(len(batch[0]))
            images, masks = batch
            #images, masks = preprocess(images, masks)
            images, masks = images.to(device), masks.to(device)
            #masks = masks.squeeze(1)  # Shape: [64, 128, 128, 2]

            #masks = masks.permute(0, 3, 1, 2)  # Shape: [64, 2, 128, 128]
            optimizer.zero_grad()
            outputs = model(images)
            #outputs = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
    
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        #print("Here",len(outputs))
        # Save the model and output masks every save_interval epochs
        if (epoch) % save_interval == 0:
            model_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'Model checkpoint saved at {model_path}')
            for j, output in enumerate(outputs):
            # Save as tensor
                # torch.save(output, os.path.join("outputs", f"output_tensor_epoch_{epoch+1}_image_{j+1}.pt"))
                
                # Convert to image (if needed)
                output = torch.argmax(output, dim=0)
                output_image = output.cpu().detach().numpy()

                output_image = (output_image).astype(np.uint8) * 255
                output_image = output_image.astype(np.uint8)
                #Image.fromarray(output_image).save(os.path.join("output2", f"output_image_epoch_{epoch+1}_image_{j+1}.png"))
        
            # model.eval()
            # with torch.no_grad():
            #     for i, (images, masks) in enumerate(train_loader):
                    
            #         images = images.to(device)
            #         outputs = model(images)
            #         predicted_masks = torch.argmax(outputs, dim=1)
            #         for j in range(images.size(0)):
            #             mask_path = f'output_masks/epoch_{epoch+1}_image_{i*images.size(0)+j}.png'
            #             os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            #             save_image(predicted_masks[j], mask_path)
            #             print(f'Saved mask {mask_path}')
    
    torch.save(model,f"seg_inter/{cat}.pth")
    print('Training complete.')


