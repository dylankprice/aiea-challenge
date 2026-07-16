import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import random_split
from torch.utils.data import DataLoader



#V2X-Real class num
num_classes = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.001
epochs = 10
train_split = 0.7
valid_split = 0.2

#input lidar data
dataset = None

train_size = len(dataset) * train_split
valid_size = len(dataset) * valid_split
test_size = len(dataset) - train_size - valid_size
dataset_train, datasest_valid, dataset_test = random_split(dataset, [train_size, valid_size, test_size])



data_loader = DataLoader(dataset_train, 
                         batch_size=2, 
                         shuffle=True)

data_loader_test = DataLoader(dataset_test,
                              batch_size=1,
                              shuffle=False)

#input second network model
model = None
model.to(device)
params = model.parameters()



optimizer = torch.optim.Adam(params, lr=lr)



#SECOND loss function

#directional loss function 
def yaw_loss(pred_yaw, target_yaw):
    angle_diff = torch.sin(pred_yaw - target_yaw)
    zeros = torch.zeros_like(angle_diff)
    return nn.SmoothL1Loss()(angle_diff, zeros)

def focal_loss(pred_class_logits, target_classes, alpha = 0.25, gamma = 2.0):
    #pass in logits, built in sigmoid
    #gamma balances loss for hard and easy examples
    #alpha balances loss for positive and negative examples
    
    return sigmoid_focal_loss(pred_class_logits, target_classes, alpha = alpha, gamma = gamma, reduction='mean')

def direction_loss(pred_dirs, target_dirs):
    #gives direction using softmax loss fn
    return nn.CrossEntropyLoss()(pred_dirs, target_dirs)

def total_loss(pred_class_logits, target_classes, pred_yaw, target_yaw, pred_dirs, target_dirs, alpha=0.25, gamma=2.0, class_weight=1.0, reg_weight=2.0, dir_weight=0.1):
    class_loss = class_weight * focal_loss(pred_class_logits, target_classes, alpha, gamma)
    reg_loss = reg_weight * yaw_loss(pred_yaw, target_yaw)
    dir_loss = dir_weight * direction_loss(pred_dirs, target_dirs)

    return class_loss + reg_loss + dir_loss

#adjust total loss for position loss too, L_reg - other

def train_step(model = model, 
               data_loader = data_loader, 
               optimizer = optimizer,
               loss_fn = loss_total, 
               device = device):
    
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        #fix dataloader unpacking, adapted for LiDAR data

        y_pred = model(X)

        #ADD: unpacking of predictions for loss arguments

        loss = loss_fn() #ADD: Loss arugments

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_step(model = model,
              data_loader = data_loader_test,
              loss_fn = loss_total,
              device = device):
    
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            #ADD: unpacking of predictions for loss arguments

            test_loss = loss_fn()  #ADD: Loss arugments

    return test_loss.item()



def train(model = model,
            data_loader = data_loader,
            data_loader_test = data_loader_test,
            optimizer = optimizer,
            loss_fn = loss_total,
            epochs = epochs,
            device = device):

    best_test_loss = torch.inf

    for epoch in range(epochs):

        print(f"Epoch {epoch+1} of {epochs}")

        train_step(model, data_loader, optimizer, loss_fn, device)

        test_loss = test_step(model, data_loader_test, loss_fn, device)



        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_weights")