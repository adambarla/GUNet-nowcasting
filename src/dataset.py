import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from datetime import datetime, timedelta
from functools import reduce
from sklearn.model_selection import train_test_split


def __unchunk__(arr):
    return reduce(lambda x,y: x+y,arr)

def get_images_like_phydnet(path, 
               stride_minutes, 
               len_in, 
               len_out,
               chunk_size,
               test_frac,
               val_frac, 
               seed):
    stride = timedelta(minutes=stride_minutes)
    timeformat='%y%m%d%H%M.png'
    window_stride = 10

    dt2file = lambda x: os.path.join(path, x.strftime(timeformat))

    datetimes = [datetime.strptime(x, timeformat) for x in os.listdir(path) if x.endswith('.png')]

    input_chunks = []
    target_chunks = []

    # Generate sequences using a sliding window
    for start_datetime in datetimes[::window_stride]: 
        input_files = []
        target_files = []
        for i in range(1, len_in + len_out + 1):
            file = dt2file(start_datetime + stride * i)
            if os.path.exists(file):
                if i <= len_in:
                    input_files.append(file)
                else:
                    target_files.append(file)
            else:
                break
        # Only if all files in sequence exist - write
        else:
            input_chunks.append(input_files)
            target_chunks.append(target_files)
            
    testval_frac = test_frac + val_frac
    X_train, X_valtest, y_train, y_valtest = train_test_split(input_chunks, 
                                                          target_chunks, 
                                                          test_size=testval_frac, 
                                                          random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, 
                                                    y_valtest, 
                                                    test_size=test_frac/(testval_frac), 
                                                    random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_images(path, 
               stride_minutes, 
               len_in, 
               len_out,
               chunk_size,
               test_frac,
               val_frac, 
               seed):
     
    stride = timedelta(minutes=stride_minutes)
    timeformat='%y%m%d%H%M.png'

    dt2file = lambda x: os.path.join(path, x.strftime(timeformat))

    datetimes = [datetime.strptime(x, timeformat) for x in os.listdir(path) if x.endswith('.png')]

    input_chunks = []
    target_chunks = []
    cnt = 0
    for start_datetime in datetimes[::chunk_size]: 
        input_chunk = []
        target_chunk = []
        for offset in range(0, chunk_size - (len_in + len_out) + 1):
            input_files = []
            target_files = []
            for i in range(1 , len_in + len_out + 1):
                file = dt2file(start_datetime + stride * (i + offset))
                if os.path.exists(file):
                    if i <= len_in:
                        input_files.append(file)
                    else:
                        target_files.append(file)
                else:
                    break
            else:
                input_chunk.append(input_files)
                target_chunk.append(target_files)

        input_chunks.append(input_chunk)
        target_chunks.append(target_chunk)
    
    
    testval_frac = test_frac + val_frac
    X_train, X_valtest, y_train, y_valtest = train_test_split(input_chunks, 
                                                              target_chunks, 
                                                              test_size=testval_frac, 
                                                              random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, 
                                                    y_valtest, 
                                                    test_size=test_frac/(testval_frac), 
                                                    random_state=seed)


    X_train = __unchunk__(X_train)
    y_train = __unchunk__(y_train)
    X_val = __unchunk__(X_val)
    y_val = __unchunk__(y_val)
    X_test = __unchunk__(X_test)
    y_test = __unchunk__(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

class CustomDataset(Dataset):
    
    def __init__(self, inputs, targets, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_images = []
        target_images = []
        
        for path in self.inputs[idx]:
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)
            input_images.append(image)
            
        for path in self.targets[idx]:
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)
            target_images.append(image)
        return [torch.cat(input_images, dim=0),torch.cat(target_images, dim=0),]

   
    
