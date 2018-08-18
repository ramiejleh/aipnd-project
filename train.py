import argparse
import torch
from torch import nn
from torch import optim
# Import helper function
from preparation_functions import load_datasets, load_classes_json, process_image, imshow
from model_functions import create_model, train_network, validate_model, save_checkpoint, load_checkpoint, predict


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('dir', type=str, default='flower_data',
                        help='path to dataset')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='chosen model: vgg or densenet')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, nargs=2, default=[4096,2048],
                        help='specify hidden units')
    parser.add_argument('--epochs', type=int, default=5,
                        help='epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='path to save the checkpoint')
    parser.add_argument('--gpu', type=bool, default=False,
                        help='use gpu?')

    return parser.parse_args()


arguments = get_input_args()

if arguments.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

#load datasets
loaded_datasets = load_datasets(arguments.dir)

#create the model
model = create_model(loaded_datasets['image_datasets'], arguments.arch, arguments.hidden_units)

#train the model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arguments.learning_rate)
train_network(loaded_datasets['dataloaders'], model, criterion, optimizer, arguments.epochs, device)

#save trained model
save_checkpoint(criterion, arguments.epochs, optimizer, model, arguments.arch, arguments.save_dir)

# print("Command Line Arguments:\n    dir =", arguments.dir,
#       "\n    arch =", arguments.arch,
#       "\n    learning_rate =", arguments.learning_rate,
#       "\n    hidden_units =", arguments.hidden_units,
#       "\n    gpu =", arguments.gpu,
#       "\n    epochs =", arguments.epochs,
#       "\n    save_dir =", arguments.save_dir)