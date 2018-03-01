import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas

from embegym.utils import get_tqdm


def ensure_var(obj):
    if isinstance(obj, (list, tuple)):
        return [ensure_var(elem) for elem in obj]
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def mcuda(obj, cuda):
    if isinstance(obj, (list, tuple)):
        return [mcuda(elem, cuda) for elem in obj]
    return obj.cuda() if cuda else obj.cpu()


def run_network_over_data(data, network, criterion, optimizer=None,
                          metrics={}, verbose=1, max_batches=None, clip_gradients=None):
    if verbose > 0:
        data = get_tqdm(data)

    cuda = next(iter(network.parameters())).data.is_cuda

    metric_values = []

    for i, (x, y) in enumerate(data):
        if max_batches is not None and i >= max_batches:
            break

        x, y = mcuda(ensure_var((x, y)), cuda)

        cur_out = network(x)
        loss = criterion(cur_out, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            if not clip_gradients is None:
                nn.utils.clip_grad_norm(network.parameters(), clip_gradients)
            optimizer.step()

        cur_metrics = {'loss': loss.data[0]}
        cur_metrics.update((name, func(cur_out, y))
                           for name, func in metrics.items())
        metric_values.append(cur_metrics)

    return pandas.DataFrame(metric_values).mean(axis=0).to_dict()


def np2tensor(arr, cuda=False):
    return mcuda(torch.from_numpy(arr), cuda)


def module_on_cuda(m):
    try:
        return next(iter(m.parameters())).data.is_cuda
    except StopIteration:
        return False
