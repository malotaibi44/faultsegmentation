from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc
import pandas as pd 

import torch
from torch.utils.data import Dataset
from skimage.metrics import hausdorff_distance
from medpy.metric.binary import dc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms , models
import os
from PIL import Image
from tqdm import tqdm
# Define a simple function to save masks
#from dataloaders import make_dataloader

# Example usage
torch.cuda.empty_cache()



import torch
from torch import nn, optim
from torchvision.models.segmentation import fcn_resnet50
import matplotlib.pyplot as plt

# Import the make_dataloader function from dataloaders.py
import os 
# Assuming you have a configuration object 'cfg' similar to what your dataloaders.py expects
from cfg import _c as cfg
from PIL import Image
import numpy as np 



# Initialize the dataloaders
import os
import torch
from torch.utils.data import DataLoader
fp='/home/hice1/malotaibi44/scratch/segformer/'
# from f3fault import F3Section
from cfg import _c as cfg
from f3sec import F3Sec
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
        dataset = F3Sec(cfg, trainlist, 'train',classannot)
    else:
        dataset = F3Sec(cfg, testlist, 'test',classannot) 
    # img,lbl=dataset.__getitem__(0)
    return dataset

def make_dataloader(cfg, subset, classannot):
    # create all dataloaders
    dataset = make_dataset(cfg,subset, classannot)
    
    loader = DataLoader(dataset = dataset,
                              batch_size = 64,
                              num_workers = cfg.dataloader.num_workers,
                              shuffle = True
                             )

    # test_loader = DataLoader(dataset = test_dataset,
    #                         batch_size = cfg.test.batch_size,
    #                         num_workers = cfg.dataloader.num_workers,
    #                         shuffle = False
    #                        ) 


    return loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model (assuming binary segmentation for simplicity, adjust as necessary)
num_classes=2
# Initialize the model (assuming binary segmentation for simplicity, adjust as necessary)
model = fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)  # Adjust num_classes based on your dataset
model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)  # Adjust for binary segmentation
model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust aux classifier
model.to(device)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            # Forward pass
            #print((masks.shape))
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")
        if (epoch) % 10 == 0:
            for j, output in enumerate(outputs):
            # Save as tensor
                # torch.save(output, os.path.join("outputs", f"output_tensor_epoch_{epoch+1}_image_{j+1}.pt"))
                output=torch.argmax(output, dim=1)
                # Convert to image (if needed)
                output_image = output.cpu().detach().numpy()
                #output_image = np.transpose(output_image, (1, 2, 0))  # Convert CxHxW to HxWxC if needed
                output_image = (output_image > 0.5).astype(np.uint8) * 255
                output_image = output_image.astype(np.uint8)
                #Image.fromarray(output_image).save(os.path.join("outputdeeplab", f"output_image_epoch_{epoch+1}_image_{j+1}.png"))
            
# Train the model
for i in range(1,28):
    cat='novice'+"{:02}".format(i)
    train_loader = make_dataloader(cfg,'train',[cat])
    # Initialize the model (assuming binary segmentation for simplicity, adjust as necessary)
    model = fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)  # Adjust num_classes based on your dataset
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)  # Adjust for binary segmentation
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust aux classifier
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_loader, criterion, optimizer)
    #expert_loader=make_dataloader(cfg,'test',['expert'])
    #expert_results = evaluate_model(model, expert_loader, device)
    #results_dict['Model'].append(cat)
    #results_dict['mIoU'].append(expert_results['mIoU'])
    #results_dict['F1 Score'].append(expert_results['F1 Score'])
    #results_dict['Hausdorff Distance'].append(expert_results['Hausdorff Distance'])
    #results_dict['Dice Score'].append(expert_results['Dice Score'])
    # Convert the results dictionary to a DataFrame
    #results_df = pd.DataFrame(results_dict)
    
    # Write the DataFrame to an Excel sheet

    
    #print("expert Results:", expert_results) 
    torch.save(model, f"fcn_models/fcn_{cat}.pth")
    
    
    
