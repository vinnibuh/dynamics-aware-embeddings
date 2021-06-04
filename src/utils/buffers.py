import numpy as np
import os
import torch
from torch.utils.data import Dataset


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage.pop(0)
        self.storage.append(data)

    # @profile
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        # X, Y, U, R, D = self.storage[ind[0]]
        # x = np.empty((batch_size, *X.shape))
        # y = np.empty((batch_size, *Y.shape))
        # u = np.empty((batch_size, *U.shape))
        # r = np.empty((batch_size, 1))
        # d = np.empty((batch_size, 1))

        # for i, index in enumerate(ind):
        #     X, Y, U, R, D = self.storage[index]
        #     x[i] = X
        #     y[i] = Y
        #     u[i] = U
        #     r[i] = R
        #     d[i] = D
        # result = (x, y, u, r, d)

        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        # import ipdb; ipdb.set_trace()
        result = (np.stack(x),
                  np.stack(y),
                  np.stack(u),
                  np.stack(r).reshape(-1, 1),
                  np.stack(d).reshape(-1, 1))
        return result

    def __len__(self):
        return len(self.storage)

    def sample_seq(self, batch_size, seq_len):
        ind = np.random.randint(0, len(self.storage) - seq_len + 1, size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            transition_sequence = self.storage[i:i + seq_len]
            # take the sequence [(xyurd), (xyurd), (xyurd), (xyurd)]
            # and turn it into [(xxxx), (yyyy), (uuuu), (rrrr), (dddd)]
            X, Y, U, R, D = list(zip(*transition_sequence))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            # import ipdb; ipdb.set_trace()
            d.append(np.array(D, copy=False))

        # import ipdb; ipdb.set_trace()
        result = (np.stack(x),
                  np.stack(y),
                  np.stack(u),
                  np.stack(r).reshape(batch_size, seq_len),
                  np.stack(d).reshape(batch_size, seq_len))
        return result


class ReplayDataset(Dataset):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage.pop(0)
        self.storage.append((
            data[0].astype('float32'),
            data[1].astype('float32'),
            data[2].astype('float32'),
            data[3],
            data[4]))

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        stacked_arrays = [np.stack([d[i] for d in self.storage]) for i in range(5)]
        manifest = {
            'type': 'float32',
            'shapes': [array.shape for array in stacked_arrays]
        }
        torch.save(manifest, "{}/manifest.pt".format(path))

        saved_arrays = [
            np.memmap("{}/{}.npmm".format(path, i), dtype='float32', mode='write', shape=stacked_arrays[i].shape)
            for i in range(5)]
        for saved_array, stacked_array in zip(saved_arrays, stacked_arrays):
            saved_array[:] = stacked_array[:]
            saved_array.flush()

    def load(self, path):
        manifest = torch.load("{}/manifest.pt".format(path))
        saved_arrays = [
            np.memmap("{}/{}.npmm".format(path, i), mode='r', dtype=manifest['type'], shape=manifest['shapes'][i])
            for i in range(5)]
        self.storage = [tuple((np.array(saved_array[i]) for saved_array in saved_arrays))
                        for i in range(saved_arrays[0].shape[0])]
        self.storage = list(self.storage)

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, i):
        return self.storage[i]


class DiskReplayDataset(Dataset):
    def __init__(self, path, max_size=1e6):
        self.storages = None
        self.path = path
        self.max_size = int(max_size)

        self.pointer = 0
        # self.size = 0

    def create_disk_arrays(self, data):
        def type_for(d):
            d_type = np.array(d).dtype
            if 'float' in str(d_type): d_type = 'float32'
            return d_type

        def shape_for(d):
            return (self.max_size,) + d.shape if isinstance(d, np.ndarray) else (self.max_size,)

        types = [type_for(d) for d in data]
        shapes = [shape_for(d) for d in data]
        os.makedirs(self.path, exist_ok=True)
        manifest = {
            'types': types,
            'shapes': shapes
        }
        torch.save(manifest, "{}/manifest.pt".format(self.path))
        self.storages = [np.memmap("{}/{}.npmm".format(self.path, i), dtype=types[i], mode='write', shape=shapes[i])
                         for i in range(len(data))]

    def add(self, data):
        if self.storages is None:
            def shape(d): return d.shape if isinstance(d, np.ndarray) else (1,)

            # import ipdb; ipdb.set_trace()
            sizes = tuple([(self.max_size,) + shape(data_elem) for data_elem in data])
            self.create_disk_arrays(data)
        address = self.pointer % self.max_size
        for storage, data_elem in zip(self.storages, data):
            storage[address] = data_elem
        self.pointer += 1

    def save(self, path):
        manifest = {
            'types': [s.dtype for s in self.storages],
            'shapes': [s.shape for s in self.storages],
            'pointer': self.pointer,
        }
        torch.save(manifest, "{}/manifest.pt".format(self.path))
        for storage in self.storages:
            storage.flush()

    def load(self, path):
        manifest = torch.load("{}/manifest.pt".format(path))
        self.storages = [np.memmap("{}/{}.npmm".format(path, i), mode='r',
                                   dtype=manifest['types'][i],
                                   shape=manifest['shapes'][i])
                         for i in range(5)]
        self.pointer = manifest['pointer']

    def __len__(self):
        return min(self.max_size, self.pointer)

    def __getitem__(self, i):
        return tuple((np.array(storage[i]) for storage in self.storages))


class EmbeddedReplayDataset(Dataset):
    def __init__(self, max_size=1e6, traj_len=4):
        self.storage = []
        self.max_size = max_size
        self.traj_len = traj_len

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage.pop(0)
        self.storage.append((
            data[0].astype('float32'),
            data[1].astype('float32'),
            data[2].astype('float32'),
            data[3].astype('float32'),
            data[4],
            data[5],
            data[6]))

    def __len__(self):
        return len(self.storage) - self.traj_len

    def __getitem__(self, i):
        transition_sequence = self.storage[i:i + self.traj_len]
        # take the sequence [(xyurd), (xyurd), (xyurd), (xyurd)]
        # and turn it into [(xxxx), (yyyy), (uuuu), (rrrr), (dddd)]
        X, Y, U, E, I, R, D = list(zip(*transition_sequence))
        result = (
            np.array(X, copy=False),
            np.array(Y, copy=False),
            np.array(U, copy=False),
            np.array(E, copy=False),
            np.array(I, copy=False),  # .reshape(-1, 1),
            np.array(R, copy=False),  # .reshape(-1, 1),
            np.array(D, copy=False))  # .reshape(-1, 1))
        # for r in result: print(r.shape)
        # import ipdb; ipdb.set_trace()
        return result


# Expects tuples of (state, next_state, action, embedded_plan, plan_step, reward, done)
class EmbeddedReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage.pop(0)
        self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, e, i, r, d = [], [], [], [], [], [], []

        for j in ind:
            X, Y, U, E, I, R, D = self.storage[j]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            e.append(np.array(E, copy=False))
            i.append(np.array(I, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        result = (np.array(x),
                  np.array(y),
                  np.array(u),
                  np.array(e),
                  np.array(i).reshape(-1, 1),
                  np.array(r).reshape(-1, 1),
                  np.array(d).reshape(-1, 1))
        return result

    def __len__(self):
        return len(self.storage)

    def sample_seq(self, batch_size, seq_len):
        ind = np.random.randint(0, len(self.storage) - seq_len + 1, size=batch_size)
        x, y, u, e, i, r, d = [], [], [], [], [], [], []

        for j in ind:
            transition_sequence = self.storage[j:j + seq_len]
            # take the sequence [(xyurd), (xyurd), (xyurd), (xyurd)]
            # and turn it into [(xxxx), (yyyy), (uuuu), (rrrr), (dddd)]
            X, Y, U, E, I, R, D = list(zip(*transition_sequence))
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            e.append(np.array(E, copy=False))
            i.append(np.array(I, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        result = (np.stack(x),
                  np.stack(y),
                  np.stack(u),
                  np.stack(e),
                  np.stack(i).reshape(batch_size, seq_len),
                  np.stack(r).reshape(batch_size, seq_len),
                  np.stack(d).reshape(batch_size, seq_len))
        return result
