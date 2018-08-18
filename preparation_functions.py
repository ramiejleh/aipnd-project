import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

def load_datasets(data_directory):
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'
    # Done: Define your transforms for the training, validation, and testing sets
    data_transforms = OrderedDict([
        ('train', transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])),
        ('test', transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]))
    ])

    # Done: Load the datasets with ImageFolder
    image_datasets = OrderedDict([
        ('train', datasets.ImageFolder(train_dir, transform=data_transforms['train'])),
        ('test', datasets.ImageFolder(test_dir, transform=data_transforms['test'])),
        ('validate', datasets.ImageFolder(valid_dir, transform=data_transforms['test']))
    ])

    # Done: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = OrderedDict([
        ('train', torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)),
        ('test', torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)),
        ('validate', torch.utils.data.DataLoader(image_datasets['validate'], batch_size=64, shuffle=True))
    ])

    loaded_datasets = OrderedDict([
        ('dataloaders', dataloaders),
        ('image_datasets', image_datasets)
    ])
    return loaded_datasets

def load_classes_json(file_path):
    with open(file_path, 'r') as f:
        mapping_of_categories  = json.load(f)
    return mapping_of_categories
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Done: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image).convert('RGB')
    width, height = pil_image.size
    if height > width:
        ratio = float(width) / float(height)
        new_width = ratio * 256
        pil_image = pil_image.resize((int(floor(new_width)), 256), Image.ANTIALIAS)
    elif width > height:
        ratio = float(height) / float(width)
        new_height = ratio * 256
        pil_image = pil_image.resize((256, int(floor(new_height))), Image.ANTIALIAS)
    else:
        pil_image.thumbnail((256,256))
    width, height = pil_image.size
    new_width = 224
    new_height= 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    cropped_pil_image = pil_image.crop((left, top, right, bottom))
    np_image = np.array(cropped_pil_image)/255
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - means)/stds

    image = normalized_image.transpose((2, 0, 1))
    return image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax