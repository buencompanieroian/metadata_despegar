#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import numpy as np
import torchvision
import torchvision.utils as vutils

from PIL import ImageFile
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms
from torchvision.transforms import ToPILImage
from sklearn.metrics import balanced_accuracy_score
from IPython.display import Image as ShowIMG
from model.densenet import densenet201
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize model 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):

    model_ft = densenet201(pretrained=use_pretrained,progress=False) 
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
    input_size = 224
    return model_ft, input_size

# Save answer file by classifying all images in test
def evaluateVal(model,dataloader,stage):
    model.eval()
    print('==> Total samples in {} set:'.format(stage),len(dataloader))
    print('==> Classifiying...')
    test_results = np.zeros((len(dataloader),2),np.int)
    cat_total   = np.zeros((16),np.int)
    cat_correct = np.zeros((16),np.int)
    
    i = 0  
    img = 0

    with torch.set_grad_enabled(False):
        for inputs, labels in dataloader: 
            # Avoid missing files
            if stage == 'test_first_stage':
                if img == 4856 or img == 9046: # missing images in first test set
                    img = img + 1
            elif stage == 'test_second_stage':
                if img == 1213 or img == 3574 or img == 6086: # missing images in second test set
                    img = img + 1
            else:
                print('==> WARNING! Unexpected test set provided')
            
            # Predict
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predict = preds.cpu().numpy()[0]
            
            # Add to numpy array
            test_results[i,0] = int(img)
            test_results[i,1] = int(predict)
            
            if i%500 == 0:
                print("==> {}%".format(int((i*100)/len(dataloader))))
		
            
            i = i+1
            img = img + 1
            
    print('Done!')
    np.savetxt("./solution_{}.csv".format(stage), test_results, fmt='%i',delimiter=",") 
    print('Saved at solution_{}.csv'.format(stage))

# path to the ImageFolder structure
data_dir = "./data"

# Number of classes in the dataset
num_classes = 16

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

# Initialize the model 
print("==> Initializing Model...")
model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
print("==> Done!")

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
print("==> Model will be run in {} device".format(device))

print("==> Initializing Datasets and Dataloaders...")
# Transform for dataloaders
data_transforms = {
    'test_first_stage': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_second_stage': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create test datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test_first_stage', 'test_second_stage']}

# Create test dataloaders
dataloaders_dict = {}
dataloaders_dict['test_first_stage']  = torch.utils.data.DataLoader(image_datasets['test_first_stage'], batch_size=1,  shuffle=False, num_workers=4)
dataloaders_dict['test_second_stage'] = torch.utils.data.DataLoader(image_datasets['test_second_stage'], batch_size=1, shuffle=False, num_workers=4)

print("==> Done!")

# Compute solutions for first and second dataset
model_ft.load_state_dict(torch.load("./checkpoint/checkpoint_densenet.ckpt"))
evaluateVal(model_ft, dataloaders_dict['test_first_stage'], 'test_first_stage')
evaluateVal(model_ft, dataloaders_dict['test_second_stage'],'test_second_stage')

print("==> Exit")

