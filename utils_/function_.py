import torch,math
from torch import optim
import numpy as np
from PIL import Image
from scipy.spatial import distance
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, meta in train_loader:
        # Moving inputs and targets to the correct device
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)
        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets,meta in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(val_loader)


def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer
def get_lr_scheduler(optimizer, cfg):
    if cfg['method'] == 'reduce_plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg['reduce_plateau_patience'],
            factor=cfg['reduce_plateau_factor'],
            cooldown=cfg['cooldown'],
            verbose=False
        )
    elif cfg['method'] == 'cosine_annealing':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['cosine_annealing_T_max'])
    elif cfg['method'] == 'constant':
        lr_scheduler = None  # No learning rate scheduling for constant LR
    else:
        raise ValueError("Invalid learning rate scheduling method")
    
    return lr_scheduler


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

def visualize_and_save_landmarks(image_path, 
                                 preds, maxvals, save_path,text=False):
    print(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure preds and maxvals are NumPy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.squeeze(0).numpy()
    if isinstance(maxvals, torch.Tensor):
        maxvals = maxvals.squeeze().numpy()
    # Draw landmarks on the image
    cnt=1
    for pred, maxval in zip(preds, maxvals):
        x, y = pred
        # x,y=x*w_r,y*h_r
        cv2.circle(img, (int(x), int(y)), 8, (255, 0, 0), -1)
        if text:
            cv2.putText(img, f"{maxval:.2f}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"{cnt}", (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cnt+=1
    # Save the image with landmarks
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return preds,maxvals
class lr_sche():
    def __init__(self,config):
        self.warmup_epochs=config["warmup_epochs"]
        self.lr=config["lr"]
        self.min_lr=config["min_lr"]
        self.epochs=config['epochs']
    def adjust_learning_rate(self,optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr  - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr