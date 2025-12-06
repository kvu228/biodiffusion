#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UNIMIB.py

PyTorch dataloaders for UNIMIB SHAR dataset (Accelerometer signals)

Author: Based on MITBIH.py structure
Date: 2024

UNIMIB Dataset Characteristics:
- 3 channels (x, y, z accelerometer)
- 9 classes
- Sequence length: 128
- Train: 6,055 samples
- Test: 1,524 samples
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import random
import warnings
warnings.filterwarnings("ignore")

# Class names for UNIMIB dataset (9 classes)
cls_dict = {
    'StandingUpFS': 0,
    'StandingUpFL': 1,
    'Walking': 2,
    'Running': 3,
    'GoingUpS': 4,
    'Jumping': 5,
    'GoingDownS': 6,
    'LyingDownFS': 7,
    'SittingDown': 8
}
reverse_cls_dict = {v: k for k, v in cls_dict.items()}


class unimib_oneClass(Dataset):
    """
    Load one class data from UNIMIB dataset.
    Example Usage:
        class0 = unimib_oneClass(class_id=0)
    """
    def __init__(self, filename='./unimib_train.csv', reshape=True, class_id=0):
        data_pd = pd.read_csv(filename)
        
        # Filter by class (label column, convert 1-9 to 0-8 if needed)
        # Assuming labels in CSV are 1-9, convert to 0-8
        if data_pd['label'].min() > 0:
            data_pd['label'] = data_pd['label'] - 1  # Convert 1-9 to 0-8
        
        data = data_pd[data_pd['label'] == class_id]
        
        # Group by ID to create samples
        grouped = data.groupby('ID')
        samples = []
        labels_list = []
        
        for id_val, group in grouped:
            # Extract ax, ay, az channels
            ax = group['ax'].values
            ay = group['ay'].values
            az = group['az'].values
            
            # Ensure sequence length is 128
            seq_len = len(ax)
            if seq_len >= 128:
                # Take first 128 timesteps
                ax = ax[:128]
                ay = ay[:128]
                az = az[:128]
            else:
                # Pad with zeros if shorter
                ax = np.pad(ax, (0, 128 - seq_len), mode='constant')
                ay = np.pad(ay, (0, 128 - seq_len), mode='constant')
                az = np.pad(az, (0, 128 - seq_len), mode='constant')
            
            # Stack into (3, 128) shape
            sample = np.stack([ax, ay, az], axis=0)  # Shape: (3, 128)
            samples.append(sample)
            labels_list.append(class_id)
        
        self.data = np.array(samples)  # Shape: (n_samples, 3, 128)
        self.labels = np.array(labels_list)
        
        print(f'Data shape of {reverse_cls_dict[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class unimib_allClass(Dataset):
    """
    Load all 9 classes from UNIMIB dataset.
    """
    def __init__(self, filename='./unimib_train.csv', isBalanced=True, n_samples=2000, oneD=True):
        data_train = pd.read_csv(filename)
        
        # Convert labels from 1-9 to 0-8 if needed
        if data_train['label'].min() > 0:
            data_train['label'] = data_train['label'] - 1
        
        # Extract data for each class
        self.data_classes = []
        for class_id in range(9):
            class_data = data_train[data_train['label'] == class_id]
            self.data_classes.append(class_data)
        
        if isBalanced:
            # Resample each class to n_samples
            resampled_classes = []
            for class_data in self.data_classes:
                if len(class_data) > 0:
                    # Group by ID first
                    grouped = class_data.groupby('ID')
                    ids = list(grouped.groups.keys())
                    
                    # Resample IDs
                    if len(ids) >= n_samples:
                        sampled_ids = np.random.choice(ids, size=n_samples, replace=False)
                    else:
                        sampled_ids = np.random.choice(ids, size=n_samples, replace=True)
                    
                    resampled_data = []
                    for id_val in sampled_ids:
                        group = grouped.get_group(id_val)
                        resampled_data.append(group)
                    
                    resampled = pd.concat(resampled_data, ignore_index=True)
                    resampled_classes.append(resampled)
                else:
                    resampled_classes.append(class_data)
            
            train_dataset = pd.concat(resampled_classes, ignore_index=True)
        else:
            train_dataset = pd.concat(self.data_classes, ignore_index=True)
        
        # Group by ID to create samples
        grouped = train_dataset.groupby('ID')
        samples = []
        labels_list = []
        
        for id_val, group in grouped:
            label = group['label'].iloc[0]  # All rows in group have same label
            
            # Extract ax, ay, az channels
            ax = group['ax'].values
            ay = group['ay'].values
            az = group['az'].values
            
            # Ensure sequence length is 128
            seq_len = len(ax)
            if seq_len >= 128:
                ax = ax[:128]
                ay = ay[:128]
                az = az[:128]
            else:
                ax = np.pad(ax, (0, 128 - seq_len), mode='constant')
                ay = np.pad(ay, (0, 128 - seq_len), mode='constant')
                az = np.pad(az, (0, 128 - seq_len), mode='constant')
            
            # Stack into (3, 128) shape
            sample = np.stack([ax, ay, az], axis=0)
            samples.append(sample)
            labels_list.append(label)
        
        self.X_train = np.array(samples)  # Shape: (n_samples, 3, 128)
        self.y_train = np.array(labels_list)
        
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        if isBalanced:
            print(f'The dataset including {n_samples} samples per class')
        else:
            for i, class_data in enumerate(self.data_classes):
                unique_ids = class_data['ID'].nunique() if len(class_data) > 0 else 0
                print(f'Class {i}: {unique_ids} samples')
    
    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class unimib_masked(Dataset):
    """
    UNIMIB dataset with masked signals for imputation task.
    """
    def __init__(self, filename='./unimib_train.csv', reshape=True, class_id=0):
        data_pd = pd.read_csv(filename)
        
        # Convert labels
        if data_pd['label'].min() > 0:
            data_pd['label'] = data_pd['label'] - 1
        
        data = data_pd[data_pd['label'] == class_id]
        
        # Group by ID to create samples
        grouped = data.groupby('ID')
        samples = []
        labels_list = []
        
        for id_val, group in grouped:
            ax = group['ax'].values
            ay = group['ay'].values
            az = group['az'].values
            
            seq_len = len(ax)
            if seq_len >= 128:
                ax = ax[:128]
                ay = ay[:128]
                az = az[:128]
            else:
                ax = np.pad(ax, (0, 128 - seq_len), mode='constant')
                ay = np.pad(ay, (0, 128 - seq_len), mode='constant')
                az = np.pad(az, (0, 128 - seq_len), mode='constant')
            
            sample = np.stack([ax, ay, az], axis=0)
            samples.append(sample)
            labels_list.append(class_id)
        
        self.data = np.array(samples)  # Shape: (n_samples, 3, 128)
        self.labels = np.array(labels_list)
        
        # Create masked version (randomly mask 20 timesteps)
        self.cond_data = self.data.copy()
        n_samples = len(self.data)
        for i in range(n_samples):
            mask_indices = np.random.choice(128, size=20, replace=False)
            # Mask all 3 channels at these timesteps
            self.cond_data[i, :, mask_indices] = 0
        
        print(f'Data shape of {reverse_cls_dict[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'org_data': torch.FloatTensor(self.data[idx]),
            'cond_data': torch.FloatTensor(self.cond_data[idx]),
            'label': self.labels[idx],
            'idx': idx
        }


class unimib_denoising(Dataset):
    """
    UNIMIB dataset with noise for denoising task.
    """
    def __init__(self, dataroot='./unimib_train.csv', train_mode=True, reshape=True, class_id=0):
        if train_mode:
            data_pd = pd.read_csv(dataroot)
        else:
            # Load test data
            test_path = dataroot.replace('train', 'test')
            data_pd = pd.read_csv(test_path)
        
        # Convert labels
        if data_pd['label'].min() > 0:
            data_pd['label'] = data_pd['label'] - 1
        
        data = data_pd[data_pd['label'] == class_id]
        
        # Group by ID
        grouped = data.groupby('ID')
        samples = []
        labels_list = []
        
        for id_val, group in grouped:
            ax = group['ax'].values
            ay = group['ay'].values
            az = group['az'].values
            
            seq_len = len(ax)
            if seq_len >= 128:
                ax = ax[:128]
                ay = ay[:128]
                az = az[:128]
            else:
                ax = np.pad(ax, (0, 128 - seq_len), mode='constant')
                ay = np.pad(ay, (0, 128 - seq_len), mode='constant')
                az = np.pad(az, (0, 128 - seq_len), mode='constant')
            
            sample = np.stack([ax, ay, az], axis=0)
            samples.append(sample)
            labels_list.append(class_id)
        
        self.data = np.array(samples)  # Shape: (n_samples, 3, 128)
        self.labels = np.array(labels_list)
        
        # Add noise (thermal noise + motion artifacts)
        n, ch, seq_len = self.data.shape
        thermal_noise_level = 0.1
        thermal_noise = np.random.rand(n, ch, seq_len) * thermal_noise_level
        
        # Motion artifacts (spikes)
        num_spikes_per_signal = np.random.randint(1, 4, n)
        motion_artifacts = np.zeros_like(self.data)
        for i in range(n):
            spike_positions = np.random.randint(0, seq_len, num_spikes_per_signal[i])
            spike_amplitudes = np.random.uniform(0.3, 1, num_spikes_per_signal[i])
            for spike_pos, spike_amp in zip(spike_positions, spike_amplitudes):
                motion_artifacts[i, :, spike_pos:spike_pos+5] = spike_amp
        
        self.cond_data = self.data + thermal_noise + motion_artifacts
        print(f'Data shape of {reverse_cls_dict[class_id]} instances = {self.data.shape}')
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'ORG': torch.FloatTensor(self.data[idx]),
            'COND': torch.FloatTensor(self.cond_data[idx]),
            'Labels': self.labels[idx],
            'Index': idx
        }

