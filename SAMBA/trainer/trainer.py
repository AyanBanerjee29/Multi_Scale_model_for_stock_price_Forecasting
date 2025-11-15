# -*- coding: utf-8 -*-
"""
Trainer class for SAMBA model
"""

import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger
from utils.metrics import All_Metrics


class Trainer:
    """Trainer class for SAMBA model"""
    
    def __init__(self, model, loss, optimizer, train_loader, val_loader, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Note: test_loader is removed from init. 
        # The 'test' method will be static.
        
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        
        # Setup logging
        if os.path.isdir(args.get('log_dir')) == False and not args.get('debug'):
            os.makedirs(args.get('log_dir'), exist_ok=True)
        self.logger = get_logger(args.get('log_dir'), name=args.get('model'), debug=args.get('debug'))
        self.logger.info('Experiment log path in: {}'.format(args.get('log_dir')))
    
    def val_epoch(self, epoch, val_dataloader):
        """Validation epoch"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target
                output = self.model(data)
                loss = self.loss(output, label)
                
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss
    
    def train_epoch(self, epoch):
        """Training epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data
            label = target
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.loss(output, label)
            loss.backward()
            
            if self.args.get('grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get('max_grad_norm'))
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % self.args.get('log_step') == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()
        
        return train_epoch_loss
    
    def train(self):
        """
        Main training loop.
        Returns the best model's state_dict and the best validation loss.
        """
        best_model_state = None
        best_loss = float('inf')
        not_improved_count = 0
        
        start_time = time.time()
        
        for epoch in range(1, self.args.get('epochs') + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            # The val_loader is now used for early stopping
            val_epoch_loss = self.val_epoch(epoch, self.val_loader)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.logger.info('*********************************Current best model captured!')
            else:
                not_improved_count += 1
            
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.args.get('early_stop_patience')))
                    break
        
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        
        # Return the best model state
        return best_model_state, best_loss
    
    @staticmethod
    def test(model, args, data_loader, logger):
        """Test the model (static method)"""
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target
                output = model(data)
                
                y_true.append(label)
                y_pred.append(output)
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        # We don't calculate metrics here anymore,
        # just return the raw tensors.
        # main.py will handle scaling and metrics.
        return y_pred, y_true