#Author: Rami Ejleh
#Description: Helper function used to create, train, validate, save and load a model and use it to make prediction
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
from preparation_functions import process_image


def create_model(image_datasets, arch='vgg', hidden_layer=[4096, 2048], drop_p=0.5):
    # Done: Build and train your network
    if 'vgg' == arch:
        model = models.vgg16(pretrained=True)
        input_size=25088
    elif 'densenet' == arch:
        model = models.densenet121(pretrained=True)
        input_size=1024

    output_size = 102
    # freezing the parameters
    for param in model.parameters():
        param.requires_grad = False

    # Creating the clasiifier and replcaing the model's with it
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_layer[0])),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_layer[0], hidden_layer[1])),
                                            ('relu2', nn.ReLU()),
                                            ('fc3', nn.Linear(hidden_layer[1], output_size)),
                                            ('dropout', nn.Dropout(p=drop_p)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.class_to_idx = image_datasets['train'].class_to_idx

    return model

def train_network(dataloader, model, criterion, optimizer, epochs=5, device='cuda'):
    # Setting up hyperparameters
    print_every = 40
    steps = 0
    model.to(device)
    for e in range(epochs):
        model.train()
        running_loss = 0
        total = 0
        correct = 0
        for ii, (inputs, labels) in enumerate(dataloader['train']):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # accuracy
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validate_model(model, dataloader['validate'], device, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Training Accuracy: %d %%" % (100 * correct / total),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: %d %%" % (accuracy))

                running_loss = 0
        print('Finished Epoch!')
    print('Finished Training!')
# Done: Do validation on the test set
def validate_model(model, dataloader, device, criterion):
    correct = 0
    total = 0
    test_loss = 0
    model.to(device)
    model.float()
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        test_loss += criterion(outputs, labels).item() / len(dataloader)
        ps = torch.exp(outputs)
        _, predicted = torch.max(ps.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return test_loss, accuracy
# Done: Save the checkpoint
def save_checkpoint(criterion, epochs, optimizer, model, arch='vgg', path='checkpoint.pth'):
    checkpoint = {'state_dict': model.classifier.state_dict,
                  'class_to_idx': model.class_to_idx,
                  'criterion': criterion,
                  'epochs': epochs,
                  'optimizer': optimizer,
                  'arch': arch,
                  'classifier': model.classifier
                  }
    torch.save(checkpoint, path)
    print('checkpoint saved at: ', path)
# Done: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filePath='checkpoint.py', device='cpu'):
    checkpoint = torch.load(filePath, map_location=device)
    loaded_classifier = checkpoint['classifier']
    loaded_classifier.load_state_dict(checkpoint['state_dict'])
    model = models.vgg16(pretrained=True)
    if checkpoint.get('arch', 'vgg') == 'densenet':
        model = models.densenet121(pretrained=True)
    model.classifier = loaded_classifier
    model.class_to_idx = checkpoint['class_to_idx']
    loaded_checkpoint = OrderedDict([
        ('model', model),
        ('criterion', checkpoint['criterion']),
        ('epochs', checkpoint['epochs']),
        ('optimizer', checkpoint['optimizer'])
    ])
    return loaded_checkpoint

def predict(image_path, model, mapping_of_categories, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Done: Implement the code to predict the class from an image file
    processed_image = torch.from_numpy(process_image(image_path)).unsqueeze_(0)
    model.to(device)
    model.eval()
    model.double()
    with torch.no_grad():
        output = model.forward(processed_image)
    ps = torch.exp(output)
    probs, classes = ps.topk(topk)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes_array = classes.detach().numpy()
    probabilities = probs.detach().numpy()[0]
    classes_names = []
    probabilities_list = {}
    for item in np.nditer(classes_array):
        classes_names.append(mapping_of_categories[idx_to_class[int(item)]])
    for index in range(len(probabilities)):
        probabilities_list[classes_names[index]] = '%.3f' % (
            probabilities[index] * 100) + '%'

    return probabilities_list