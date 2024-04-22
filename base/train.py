import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.cuda                   
from typing import List, Callable, Tuple
from base.EMA import EMA

ema = EMA(model.parameters(), decay_rate=0.995, num_updates=0)

def train_model(model: nn.Module, criterion: Callable, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler, loaders: List[DataLoader], num_epochs: int = 10
               ) -> Tuple[nn.Module, List[float], List[float]]:  
    """
    Train a model on a dataset.

    Args:
      model: The model to train.
      criterion: The loss function.
      optimizer: The optimizer.
      scheduler: The learning_rate scheduler
      loaders: The training dataloader.
      num_epochs: The number of epochs to train for.

    Returns:
      A list of three elements. The first element is the model, the second element is a list of training losses, and the third element is a list of validation losses.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 20
    counter = 0
    train_loss = []
    valid_loss = []
    
    for epoch in range(num_epochs):
      print(f'Epoch {epoch}/{num_epochs - 1}')
      print('-' * 10)
                     
      # Training Phase
      model.train()
      running_corrects_T = 0
      running_loss_T = 0.0
      ns_T = 0
      train_dl = loaders['train']
        
      with tqdm(train_dl, desc=f"Training Epoch {epoch+1}") as pbar:

        for images, labels in pbar:
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()
          outp = model(images)
          _, pred = torch.max(outp, 1)

          smooth_labels = torch.full_like(labels, (1.0 - 0.9) / (3 - 1))#smoothing_value = (1.0 - smoothing_factor) / (num_classes - 1)
          smooth_labels[range(len(labels)), labels] = 0.9
          loss = criterion(outp, smooth_labels)

          loss.backward()
          optimizer.step()

          running_loss_T += loss.item() * images.size(0)
          running_corrects_T += torch.sum(pred == labels.data)
          ns_T += pred.shape[0]
                
          acc_value = running_corrects_T.double() / ns_T
          loss_value = running_loss_T/ ns_T
          metrics = {"Batch":f"Batch_{ns_T}","Train Accuracy":f"{acc_value:.3f}","Train Loss":f"{loss_value:.3f}"}
          pbar.set_postfix(metrics)
                
        # Validation Phase
        model.eval()

        ns_V = 0
        running_corrects_V = 0
        running_loss_V = 0.0
        valid_dl = loaders['val']
        
        with torch.no_grad():
          with tqdm(valid_dl, desc=f"Validation Epoch {epoch+1}") as pbar:

            for images, labels in pbar:
              images, labels = images.to(device), labels.to(device)
              optimizer.zero_grad()

              outp = model(images)
              _, pred = torch.max(outp, 1)
              val_loss = criterion(outp, labels)

              running_loss_V += val_loss.item() * images.size(0)
              ns_V += pred.shape[0]
              running_corrects_V += torch.sum(pred == labels.data)

              acc_value = running_corrects_V.double() / ns_V
              loss_value = running_loss_V/ ns_V

              metrics = {"Val Accuracy":f"{acc_value:.3f}","Val Loss":f"{loss_value:.3f}"}
              pbar.set_postfix(metrics)
                
        Train_loss = running_loss_T / len(train_dl.dataset)
        train_loss.append(Train_loss)
        Valid_loss = running_loss_V / len(valid_dl.dataset)
        valid_loss.append(Valid_loss)
        Train_acc = running_corrects_T.double() / ns_T
        Valid_acc = running_corrects_V.double() / ns_V
        
        scheduler.step()
        print(f'Train Loss: {Train_loss:.4f} Train Acc: {Train_acc:.4f} Valid Loss: {Valid_loss:.4f} Valid Acc: {Valid_acc:.4f}')
    
    #     if Valid_acc > best_acc:
    #         best_acc = Valid_acc
    #         best_model_wts = copy.deepcopy(model.state_dict())
    #         checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            
    #         counter = 0
    #     else:
    #         counter += 1
    #         if counter >= patience:
    #             print("Early stopping")
    #             break
            
            
    # print('Best accuracy {}'.format(best_acc))             
    model.load_state_dict(best_model_wts)
    return model, train_loss, valid_loss