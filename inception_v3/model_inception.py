import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from inception import inception_v3


def make_model():

    model = inception_v3(pretrained=False, aux_logits=False)
    
    state = model_zoo.load_url('https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth')
    clean_state = OrderedDict()
    # remove AuxLogits from the pretrained model
    for k in state:
        if not 'AuxLogits' in k:
            clean_state[k] = state[k]
    
    model.load_state_dict(clean_state)

    # make all params untrainable
    for p in model.parameters():
        p.requires_grad = False

    # reset the last layer
    model.fc = nn.Linear(2048, 256)

    # initialize the last layer's weights
    init.normal(model.fc.weight, std=0.01);
    init.constant(model.fc.bias, 0.0);

    # create different parameter groups
    classifier_weights = [model.fc.weight]
    classifier_biases = [model.fc.bias]

    classifier_lr = 1e-1

    # you need to tune only weight decay and momentum here
    optimizer = optim.SGD([
        {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-2},
        {'params': classifier_biases, 'lr': classifier_lr}
    ], momentum=0.9, nesterov=True)

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # move the model to gpu
    model = model.cuda()
    return model, criterion, optimizer
