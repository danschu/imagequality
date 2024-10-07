from torch import float32, no_grad, save
from torch.optim import Adam
from torch.nn import MSELoss
import torch
import numpy as np
import os
import copy


def train_epoch(model, dataloader, optimizer, device):
    loss_func = MSELoss()
    model.train()
    
    totalloss = 0
    totalaccuracy = 0
    for x, y in dataloader:
        x = x.to(device, dtype=float32)
        y = y.to(device, dtype=float32).flatten()
        
        optimizer.zero_grad()
        output = model(x)
        output = output.squeeze()*100

        loss = loss_func(output, y)
        loss.backward()
        totalloss += loss.item()
        optimizer.step()
        
        predicted_labels = output
        correct_predictions = 100-abs(predicted_labels-y)
        acc = float(correct_predictions.sum())
        
        totalaccuracy += acc
        
    return model, totalloss, totalaccuracy/len(dataloader.dataset)

def val_epoch(model, dataloader, device):
    loss_func = MSELoss()
    model.eval()
    
    totalloss = 0
    totalaccuracy = 0
    
    with no_grad():
        for x, y in dataloader:
            x = x.to(device, dtype=float32)
            y = y.to(device, dtype=float32).flatten()
            
            output = model(x)
            output = output.squeeze()*100
            
            predicted_labels = output
            totalloss += loss_func(output, y).item()
            correct_predictions = 100-abs(predicted_labels-y)
            acc = float(correct_predictions.sum())
            
            totalaccuracy += acc

    return totalloss, totalaccuracy/len(dataloader.dataset)

def train_model(model, model_output_path, train_data, val_data, learning_rate, device, num_epochs=200, weight_decay=0.001, patience=None):
    
    epoch = 0
    optimizer = Adam(model.parameters(), learning_rate, weight_decay=0.001)
    
    min_validation_loss = np.inf
    best_model_state_dict = None
    best_epoch = 0
    while True:
        epoch += 1
        print(f"Epoch: {epoch}")
        if epoch % 10 == 0:
            learning_rate *= 0.8
            optimizer.param_groups[0]['lr'] = learning_rate
        
        model, train_loss, train_accuracy = train_epoch(model, train_data, optimizer, device)
        print(f"Train loss: {train_loss:.3f} Train accuracy: {train_accuracy:.3f}")

        val_loss, val_accuracy = val_epoch(model, val_data, device)
        print(f"Val loss: {val_loss:.3f} Val accuracy: {val_accuracy:.3f}")
        
        if val_loss < min_validation_loss:
            print(f"! New model at epoch {epoch}")
            best_model_state_dict = state_dict = copy.deepcopy(model.state_dict())
            min_validation_loss = val_loss
            best_epoch = epoch
            save(best_model_state_dict, model_output_path)   
        
        if patience and epoch-best_epoch > patience:
            print(f"End training at epoch {epoch}")
            break
        
        if num_epochs == epoch: 
            print(f" End training at epoch {epoch}")
            break
        print("")