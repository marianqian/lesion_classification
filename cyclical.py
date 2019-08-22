#trying to make cyclical learning rate
#https://github.com/fastai/course-v3/blob/master/nbs/dl2/05_anneal.ipynb
import torch
import math
from torch import tensor
import matplotlib.pyplot as plt
from functools import partial

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    #pcts = tensor([0] + listify(pcts))
    pcts.insert(0, 0)
    pcts = torch.FloatTensor(pcts)
    #assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])




a = torch.arange(0, 100)
p = torch.linspace(0.01,1,100)

torch.Tensor.ndim = property(lambda x: len(x.shape))

#plt.plot(a, [sched(o) for o in p])
