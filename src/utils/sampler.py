import random
from collections import defaultdict

class SourceWeightedSampler:
    def __init__(self, indices, sources, source_weights, epoch_size):
        self.pool = defaultdict(list)
        for idx, src in zip(indices, sources):
            self.pool[src].append(idx)
        self.weights = source_weights
        self.epoch_size = epoch_size
        for k in self.pool: random.shuffle(self.pool[k])
        self.ptr = {k:0 for k in self.pool}

    def __iter__(self):
        keys = list(self.pool.keys())
        total = sum(self.weights.get(k, 0) for k in keys) or 1.0
        counts = {k: int(self.epoch_size * (self.weights.get(k,0)/total)) for k in keys}
        bag = []
        for k, c in counts.items():
            arr = self.pool[k]; ptr = self.ptr[k]
            for _ in range(max(1,c)):
                bag.append(arr[ptr % len(arr)]); ptr += 1
            self.ptr[k] = ptr
        random.shuffle(bag)
        for i in bag: yield i

    def __len__(self):
        return self.epoch_size
