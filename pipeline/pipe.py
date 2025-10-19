from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    if num_batches <= 0 or num_partitions <= 0:
        return []

    total_ticks = num_batches + num_partitions - 1
    for tick in range(total_ticks):
        cycle = []
        for part in range(num_partitions):
            batch_idx = tick - part
            if batch_idx < 0 or batch_idx >= num_batches:
                continue
            # scoot along the diagonals of Figure 3 basically
            cycle.append((batch_idx, part))
        if cycle:
            yield cycle
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)
        self._micro_batches = None

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        if len(self.partitions) == 0:
            return x

        micro_batches = list(x.split(self.split_size, dim=0))
        if len(micro_batches) == 0:
            return torch.empty_like(x).to(self.devices[-1])

        # stash inputs so compute can grab them
        self._micro_batches = [mb.to(self.devices[0]) for mb in micro_batches]
        num_parts = len(self.partitions)
        stage_slots: List[List[Optional[Tensor]]] = [
            [None] * num_parts for _ in self._micro_batches
        ]

        try:
            for schedule in _clock_cycles(len(self._micro_batches), num_parts):
                self.compute(stage_slots, schedule)
        finally:
            # clean up just in case
            self._micro_batches = None  # type: ignore

        outputs = [stage_slots[i][-1] for i in range(len(stage_slots))]
        outputs = [out for out in outputs if out is not None]
        if not outputs:
            return torch.empty(0, device=self.devices[-1])
        return torch.cat(outputs, dim=0)
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        pending = []
        for batch_idx, part_idx in schedule:
            if part_idx == 0:
                data = self._micro_batches[batch_idx]
            else:
                data = batches[batch_idx][part_idx - 1]
            if data is None:
                raise RuntimeError("Missing input for pipeline stage")
            data = data.to(devices[part_idx])

            partition = partitions[part_idx]
            def make_task(partition=partition, data=data):
                return Task(lambda: partition(data))

            task = make_task()
            self.in_queues[part_idx].put(task)
            pending.append((batch_idx, part_idx))

        for batch_idx, part_idx in pending:
            success, payload = self.out_queues[part_idx].get()
            if not success:
                exc_type, exc, tb = payload
                raise exc.with_traceback(tb)
            _, result = payload
            batches[batch_idx][part_idx] = result
        # END ASSIGN5_2_2
