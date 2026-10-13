import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Dataset_ETT_hour(Dataset):

    def __init__(self, args, root_path, flag='train', size=None, data_path='ETTh1.csv', target='OT', scale=True):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        if flag == 'train':
            self.window_stride = max(1, int(getattr(args, 'train_sample_stride', 1)))
        elif flag == 'val':
            self.window_stride = max(1, int(getattr(args, 'val_sample_stride', 1)))
        else:
            self.window_stride = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(local_fp)
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = df_raw[df_raw.columns[1:]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.window_indices = np.arange(0, max_start, self.window_stride, dtype=np.int64)

    def __getitem__(self, index):
        s_begin = int(self.window_indices[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.window_indices)

class Dataset_ETT_minute(Dataset):

    def __init__(self, args, root_path, flag='train', size=None, data_path='ETTm1.csv', target='OT', scale=True):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        if flag == 'train':
            self.window_stride = max(1, int(getattr(args, 'train_sample_stride', 1)))
        elif flag == 'val':
            self.window_stride = max(1, int(getattr(args, 'val_sample_stride', 1)))
        else:
            self.window_stride = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(local_fp)
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = df_raw[df_raw.columns[1:]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.window_indices = np.arange(0, max_start, self.window_stride, dtype=np.int64)

    def __getitem__(self, index):
        s_begin = int(self.window_indices[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.window_indices)

class Dataset_Custom(Dataset):

    def __init__(self, args, root_path, flag='train', size=None, data_path='ETTh1.csv', target='OT', scale=True):
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        if flag == 'train':
            self.window_stride = max(1, int(getattr(args, 'train_sample_stride', 1)))
        elif flag == 'val':
            self.window_stride = max(1, int(getattr(args, 'val_sample_stride', 1)))
        else:
            self.window_stride = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(local_fp)
        "\n        df_raw.columns: ['date', ...(other features), target feature]\n        "
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        df_data = df_raw[df_raw.columns[1:]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        max_start = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.window_indices = np.arange(0, max_start, self.window_stride, dtype=np.int64)

    def __getitem__(self, index):
        s_begin = int(self.window_indices[index])
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        return len(self.window_indices)
