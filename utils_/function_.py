import torch
import inspect
from torch import optim
import numpy as np
from PIL import Image
from scipy.spatial import distance
import cv2
def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets,meta in train_loader:
        if isinstance(inputs,list):
            inputs=[i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets, meta in val_loader:
            if isinstance(inputs,list):
                inputs=[i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

    accuracy = correct_predictions / total_predictions * 100
    avg_loss = running_loss / len(val_loader)
    return avg_loss, accuracy

def get_instance(module, class_name, *args, **kwargs):
    try:
        cls = getattr(module, class_name)
        instance = cls(*args, **kwargs)
        return instance
    except AttributeError:
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__]
        raise ValueError(f"{class_name} not found in the given module. Available classes: {', '.join(available_classes)}")


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def ridge2json(image_path,preds,maxvals):
    return {
        "image_path":image_path,
        "ridge_coordinate":preds,
        "score":maxvals
    }

def train_epoch_inception(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets,meta in train_loader:
        if isinstance(inputs,list):
            inputs=[i.to(device) for i in inputs]
        else:
            inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs,aux_logit = model(inputs)
        loss = loss_function(outputs, targets)+ \
            loss_function(aux_logit,targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def val_epoch_inception(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, meta in val_loader:
            if isinstance(inputs,list):
                inputs=[i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(outputs, dim=1)

            loss = loss_function(outputs, targets)

            running_loss += loss.item()

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(predicted_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(val_loader)

    return avg_loss

def sensitive_score(label,predict,data_list):
    success_cnt=0
    # print("wrong_list")
    ill_cnt=0
    cnt=0
    for i,j in zip(label,predict):
        if i>0:
            ill_cnt+=1
            if j>0:
                success_cnt+=1
        # if i!=j:
        #     print(data_list[cnt])    
        cnt+=1
    return success_cnt/ill_cnt

def crop_square(img_path, x, y, width, visual_path=None, save_path=None):
    # Open the image file
    img = Image.open(img_path)
    x,y=int(x),int(y)
    # Calculate the top left and bottom right points of the square to be cropped
    left = x - width
    top = y - width
    right = x + width
    bottom = y + width

    # Convert image to OpenCV format (numpy array) for visualization
    img_cv = np.array(img)

    # Draw a point at (x, y)
    img_cv = cv2.circle(img_cv, (x, y), radius=0, color=(0, 0, 255), thickness=-1)  # Red point

    # Convert back to PIL format and save visualized image
    if visual_path:
        visual_img = Image.fromarray(img_cv)
        visual_img.save(visual_path)

    # Crop the image and save it
    cropped_img = img.crop((left, top, right, bottom))
    if save_path:
        cropped_img.save(save_path)
    
    return cropped_img

def closest_points(ridge_list,vessel_list):
    array1=np.array(ridge_list)
    array2=np.array(vessel_list)
    # Compute the distance between each pair of points
    distances = distance.cdist(array1, array2, 'euclidean')

    # Find the indices of the pair with the smallest distance
    min_distance_index = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the points
    point1 = ridge_list[min_distance_index[0]]
    point2 = vessel_list[min_distance_index[1]]
    return point1,point2
def dispoint2list(point,points_list):
    point=np.array(point)
    points_list=np.array(points_list)
    distances = distance.cdist(point[np.newaxis, :], points_list, 'euclidean')[0]
    return distances