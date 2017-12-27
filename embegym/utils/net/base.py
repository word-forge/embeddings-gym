import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
import pandas


def ensure_var(obj):
    if isinstance(obj, Variable):
        return obj
    else:
        return Variable(obj)


def run_network_over_data(data, network, criterion, optimizer=None, metrics={}, verbose=1):
    if verbose > 0:
        data = tqdm.tqdm(data)

    cuda = network.parameters()[0].data.is_cuda

    metric_values = []

    for x, y in data:
        x = ensure_var(x)
        y = ensure_var(y)
        if cuda:
            x = x.cuda()
            y = y.cuda()
        else:
            x = x.cpu()
            y = y.cpu()

        cur_out = network(x)
        loss = criterion(cur_out, y)
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(network.parameters(), 10)
            optimizer.step()

        cur_metrics = {'loss': loss.data[0]}
        cur_metrics.update((name, func(cur_out, y))
                           for name, func in metrics.items())
        metric_values.append(cur_metrics)

    return pandas.DataFrame(metric_values).mean(axis=0).to_dict()


def mcuda(obj, cuda):
    return obj.cuda() if cuda else obj


def np2tensor(arr, cuda=False):
    return mcuda(torch.from_numpy(arr), cuda)


def module_on_cuda(m):
    return m.parameters()[0].data.is_cuda


def collect_batch(gen, batch_size):
    batch = []
    for sample in gen:
        batch.append(sample)
        if len(batch) >= batch_size:
            yield default_collate(batch)
    if len(batch) > 0:
        yield default_collate(batch)
