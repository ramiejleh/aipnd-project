#Author: Rami Ejleh
#Description: Using a checkpoint model to predict image classes
import argparse
import torch
from torch import nn
from torch import optim
# Import helper function
from preparation_functions import load_datasets, load_classes_json, process_image, imshow
from model_functions import create_model, train_network, validate_model, save_checkpoint, load_checkpoint, predict

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', type=str, default='flower2.jpg',
                        help='path to image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth',
                        help='Provide a checkpoint of a model')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='mapping of categories to real names')
    parser.add_argument('--top_k', type=int, default='5',
                        help='Number of top results to show')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='use gpu?')

    return parser.parse_args()

arguments = get_input_args()

#getting the arguments ready to use
if arguments.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

#load checkpoint
loaded_checkpoint = load_checkpoint(arguments.checkpoint, device)
criterion = loaded_checkpoint['criterion']
optimizer = loaded_checkpoint['optimizer']
epochs = loaded_checkpoint['epochs']
model = loaded_checkpoint['model']

#load class names mapping
mapping_of_categories = load_classes_json(arguments.category_names)

# #predict probs
top_outputs = predict(arguments.path, model, mapping_of_categories,arguments.top_k, device)

# print("Command Line Arguments:\n    path =", arguments.path,
#       "\n    checkpoint =", arguments.checkpoint,
#       "\n    category_names =", arguments.category_names,
#       "\n    gpu =", arguments.gpu,
#       "\n    top_k =", arguments.top_k)

print(top_outputs)
