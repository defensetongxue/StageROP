import torch
from torch.utils.data import DataLoader
from config import get_config
from utils_ import get_optimizer,stage12_Dataset as CustomDatset,lr_sche
from  models import build_model
import os
from utils_ import train_epoch,val_epoch
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

# Parse arguments
args = get_config()
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model,criterion = build_model(args.configs['model'])
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0

if os.path.isfile(args.from_checkpoint):
    print(f"loadding the exit checkpoints {args.from_checkpoint}")
    model.load_state_dict(
    torch.load(args.from_checkpoint))

# Creatr optimizer
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDatset(args.data_path,args.configs,split='train',split_name=args.split_name)
val_dataset=CustomDatset(args.data_path,args.configs,split='val',split_name=args.split_name)
test_dataset=CustomDatset(args.data_path,args.configs,split='test',split_name=args.split_name)
print(len(train_dataset),len(val_dataset),len(test_dataset))
# Create the data loaders
drop_last = False
if args.configs['model']['name'] == 'inceptionv3' \
    and len(train_dataset) % args.configs['train']['batch_size'] == 1:
    drop_last = True
    print("drop last in train loader")
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=drop_last)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader = DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("There is {} patch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")
early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs['train']['end_epoch']
best_acc=0
save_name=args.configs['model']['name']+'.pth'
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss, accuracy, auc = val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{total_epoches}," 
          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Acc: {accuracy:.6f}, Auc: {auc:.6f}" 
            f" Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}" )
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    # Early stopping
    if accuracy > best_acc:
        best_acc=accuracy
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,f"{args.split_name}_{save_name}"))
        print("Model saved as {}".format(os.path.join(args.save_dir,f"{args.split_name}_{save_name}")))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break
_,test_acc,test_auc=val_epoch(model,test_loader,criterion,device)
print(f"last epoch model: [test] : Acc: {test_acc}, Auc: {test_auc}")
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,f"{args.split_name}_{save_name}")))
model.eval()
_,test_acc,test_auc=val_epoch(model,test_loader,criterion,device)
print(f"best epoch model: [test] : Acc: {test_acc}, Auc: {test_auc}")
