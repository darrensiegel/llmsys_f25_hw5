from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        item = self.data[self.index[index]]
        if torch.is_tensor(item) and item.dim() == 0:
            return item.item()
        return item
        # END ASSIGN5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        indices = list(range(len(self.data)))
        rng.shuffle(indices)
        total_len = len(indices)
        if total_len == 0:
            self.partitions = [[] for _ in sizes]
            return

        total_weight = sum(sizes)
        
        split_indices = []
        start = 0
        
        for frac in sizes[:-1]:
            part_len = int((frac / total_weight) * total_len)
            end = min(start + part_len, total_len)
            split_indices.append(indices[start:end])
            start = end
        split_indices.append(indices[start:])  # Last partition gets the remainder
        self.partitions = split_indices
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partition` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        return Partition(self.data, self.partitions[partition])
        # END ASSIGN5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partitioner = DataPartitioner(dataset, sizes=partition_sizes)
    current_partition = partitioner.use(rank)

    if collate_fn is None:
        return current_partition

    partitioned_batch_size = max(1, batch_size // world_size)
    return DataLoader(current_partition, batch_size=partitioned_batch_size, collate_fn=collate_fn, shuffle=False)
    # END ASSIGN5_1
