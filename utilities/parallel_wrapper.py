#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
import threading
from torch import nn
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel.data_parallel import DataParallel

'''
This class defines data parallel wrappers for mdoel and criterian parallelism.
These are basically adapted from PyTroch's DataParallel wrapper'''

class DataParallelModel(DataParallel):

    def forward(self, *inputs, **kwargs):
        ''' The only difference between this and PyTorch's native implementation is that
        we do not need gather function because we will perform gathering inside Criterian
        wrapper.'''
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        return self.parallel_apply(replicas, inputs, kwargs)


class DataParallelCriteria(DataParallel):

    def forward(self, inputs, *targets, **kwargs):
        '''
        Input is already sliced, so slice only target
        '''
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = parallel_apply_criteria(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)


def parallel_apply_criteria(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # PyTorch's native implementation convert to tuple to avoid further slicing
                # Just extract the tensor out of the tuple to compute loss
                if isinstance(input, (list, tuple)):
                    input = input[0]
                if isinstance(target, (list, tuple)):
                    target = target[0]
                assert target.device == input.device
                if module.device != input.device:
                    module = module.to(input.device)
                output = module(input, target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target,
                                          kwargs, device),)
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs

