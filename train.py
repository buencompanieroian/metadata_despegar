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

# Compute weights (imbalanced multi-label classification)
def computeWeights(class_total,class_count):
    weights = []
    for count in class_count:
        w = class_total / (len(class_count)*count)
        weights.append(w)
    return weights

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # LR scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'max')
    max_balanced = 0
    balanced_lst = []
    
    for epoch in range(num_epochs):
        print('==> Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_pred = []
            all_labl = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_pred.extend(preds.cpu().numpy().tolist())     
                all_labl.extend(labels.data.cpu().numpy().tolist())    

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # compute balanced acc metric
            epoch_balanced = balanced_accuracy_score(np.array(all_labl), np.array(all_pred))
            print('==> {} Loss: {:.4f} Acc: {:.4f}  Bal: {:.4f}'.format(phase, epoch_loss, epoch_acc,epoch_balanced))
            
            if phase == 'val':
                balanced_lst.append(epoch_balanced)
                scheduler.step(epoch_balanced)
                
                # save model if outperformed history
                if max_balanced < epoch_balanced:
                    max_balanced = epoch_balanced
                    torch.save(model_ft.state_dict(), "model/checkpoint_{:.4f}.ckpt".format(max_balanced))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    
    # final stats
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('==> Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# Function to initialize DenseNet model. Class is provided in separate file
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize model (densenet201)
    model_ft = densenet201(pretrained=use_pretrained,progress=False) 
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
    input_size = 224
    return model_ft, input_size

print("==> Training Model")
print("==> Hyperparameters: ")

# path to the ImageFolder structure
data_dir = "./data"
print("==> Datadir={}".format(data_dir))

# Number of classes in the dataset
num_classes = 16
print("==> Classes={}".format(num_classes))

# Batch size for training (change depending on how much memory you have)
batch_size = 32 
print("==> Batch Size={}".format(batch_size))

# Number of epochs to train for
num_epochs = 100
print("==> Epochs={}".format(num_epochs))

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False
print("==> Retrain full model={}".format(not feature_extract))

# Classes and weights
classes = ['Bar','Bathroom','Bedroom','Breakfast','City','Dining','HotelFront',
           'HotelExterior','HotelInterior','Kitchen','Living','Lobby',
           'Natural','Pool','Recreation','Sports']

# Training images count
class_count =  np.asarray([148,1033,4387,604,477,408,749,808,715,404,429,623,624,384,230,198])
class_weights = torch.FloatTensor(np.asarray(computeWeights(np.sum(class_count), class_count)))
class_weights[2] = 0.5 # Extra weight to bedroom class, otherwise it receives a very low weight

# Initialize the model for this run
model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

print("==> Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("==> Done!")

print("==> Initializing Model...")
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated
params_to_update = model_ft.parameters()
print("==> Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

print("==> Done!")

# All parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.00001, betas=(0.5, 0.999))

# Setup the CrossEntropyLoss with the computed class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Train and evaluate
print("==> Training...")
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
print("==> Exit!")

