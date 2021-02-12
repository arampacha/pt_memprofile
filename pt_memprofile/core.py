# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['MemHooks', 'plot_log', 'plot_logs', 'memprofile', 'simple_model', 'MemProfileCallback', 'simple_dls',
           'MemStatsCallback']

# Cell
import torch
from torch import Tensor
import torch.nn as nn
import torch.cuda.amp as amp
from fastai.callback.all import Hooks, ShortEpochCallback, HookCallback, has_params
from fastai.basics import *

# Cell
def _generate_mem_hook(mem_log, idx, hook_type, experiment):
    "Hook function generator"
    def hook(m, *args):
        inp_shape = args[0][0].shape
        out_shape = args[1][0].shape if len(args)>1 else None
        if len(mem_log) == 0: call_idx = 0
        else: call_idx = mem_log[-1]["call_idx"] + 1
        mem_all = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
        max_all = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        mem_log.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(m).__name__,
            'experiment': experiment,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_reservd': mem_reserved,
            'max_all': max_all,
            'input_shape': inp_shape,
            'output_shape': out_shape,
        })
    return hook

# Cell
class MemHooks(Hooks):
    "Creates hooks for logging memory stats"
    def __init__(self, ms, name=None):
        self.hooks = []
        self.mem_log = []
        for i, m in enumerate(ms): self.register_memory_hooks(m, i, name=name)

    def register_memory_hooks(self, m, i, name=None):
        fs = {'pre':m.register_forward_pre_hook,
              'fwd':m.register_forward_hook,
              'bwd':m.register_backward_hook}
        for hook_type in ['pre', 'fwd', 'bwd']:
            self.hooks.append(fs[hook_type](_generate_mem_hook(self.mem_log, i, hook_type, name)))

# Cell
def plot_log(mem_log:pd.DataFrame):
    plt.plot(mem_log['call_idx'], mem_log['mem_all']/1024)
    plt.ylabel('Memory allocated (Kb)');

# Cell
def plot_logs(*mem_logs):
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    legend = []
    for log in mem_logs:
        plt.plot(log['mem_all']/1024)
        legend += [log.loc[0, 'experiment']]
    if len(mem_logs) > 1: ax.legend(legend)
    plt.ylabel('Memory allocated (Kb)')
    plt.show()

# Cell
def memprofile(model:nn.Module, xb:Tensor, yb:Tensor, loss_func=CrossEntropyLossFlat(), plot=True, label=None, fp16=False):
    "Records memory stats for one forward-and-backward pass through the model with batch (xb, yb)"
    def forward_and_loss():
        out = model(xb)
        return loss_func(out, yb)
    label = ifnone(label, type(model).__name__)
    device = xb.device
    model.to(device)
    prealloc = torch.cuda.memory_allocated()
    with MemHooks(flatten_model(model), label) as h:
        if fp16: forward_and_loss = amp.autocast()(forward_and_loss)
        loss = forward_and_loss()
        loss.backward()
        model.zero_grad(set_to_none=True)
        mem_log = pd.DataFrame(h.mem_log, copy=True)
    mem_log['mem_all'] = mem_log['mem_all'] - prealloc
    if plot:
        plot_log(mem_log)
    return mem_log

# Cell
def simple_model(ni=100, no=2, n=4):
    layers = [nn.Linear(ni, ni) for i in range(n)] + [nn.Linear(ni,no)]
    return nn.Sequential(*layers)

# Cell
class MemProfileCallback(Callback):
    "Cancels fit after one batch before weight update"
    def before_step(self):
        self.model.zero_grad(set_to_none=True)
        raise CancelStepException
    def after_batch(self):
        print('Fit canceled')
        raise CancelFitException

# Cell
@patch
def profile_memory(self:Learner, plot=True):
    """
    Records memory stats for single forward-and-backward pass
    """
    with MemHooks(flatten_model(self.model), type(self.model).__name__) as h:
        prealloc = torch.cuda.memory_allocated()
        with self.added_cbs(MemProfileCallback()), self.no_logging():
            self.fit(1)
        mem_log = pd.DataFrame(h.mem_log, copy=True)
    mem_log['mem_all'] = mem_log['mem_all'] - prealloc
    if plot:
        plot_log(mem_log)

    return mem_log

# Cell
def simple_dls():
    train = [(torch.randn(100), torch.randint(2, (1,))) for _ in range(800)]
    valid = [(torch.randn(100), torch.randint(2, (1,))) for _ in range(200)]
    return DataLoaders.from_dsets(train, valid, bs=16, device='cuda')

# Cell
class MemStatsCallback(HookCallback):
    "Registers memory hooks on modules in `ms`"
    def __init__(self, ms=None, label=None, remove_end=True):
        store_attr()
        self.prealloc = torch.cuda.memory_allocated()
        self.every = None

    def _register(self): self.hooks = MemHooks(self.ms, name=self.label)

    def before_fit(self):
        "Register `self.hooks` on `self.ms`."
        if self.ms is None: self.ms = [m for m in flatten_model(self.model) if has_params(m)]
        if self.every is None: self._register()

    def after_fit(self):
        self.stats = pd.DataFrame(self.hooks.mem_log, copy=True)
        self.stats['mem_all'] = self.stats['mem_all'] - self.prealloc
        if self.remove_end: self._remove()

    def plot(self): plot_log(self.stats)