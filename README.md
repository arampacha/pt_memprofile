# pt_memprofile
> Utilities for cuda memory usage inspection for pytorch and fastai.


## Installation

`pip install -qq git+git://github.com/arampacha/pt_memprofile.git`

## How to use

### Profiling TransformerLM

```python
bs, sl = 8, 128
dls = DataLoaders.from_dsets(DeterministicTwinSequence(sl, 10*bs), 
                             DeterministicTwinSequence(sl, 2*bs), 
                             bs=bs, shuffle=False, device='cuda')
model = TransformerLM(128, 256)
```

```python
xb, yb = dls.one_batch()
memlog1 = memprofile(model, xb, yb, plot=False, label='fp32')
memlog2 = memprofile(model, xb, yb, plot=False, label='fp16', fp16=True)
```

```python
plot_logs(memlog1, memlog2)
```


![png](docs/images/output_6_0.png)


### Fastai interface

```python
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())
memlog1 = learn.profile_memory()
```


![png](docs/images/output_8_0.png)


### Profiling epoch
{% include note.html content='Memory profiling slows down training significantly, so you probably don&#8217;t want to run it for a comlete epoch.' %}

Let's examine effect of optimizer to memory usage:

```python
#hide_output
#cuda
learn = Learner(simple_dls(), simple_model(), loss_func=CrossEntropyLossFlat(), cbs=MemStatsCallback(label='SGD'), opt_func=SGD)
with learn.no_bar(): learn.fit(1, 1e-3, cbs=[ShortEpochCallback(pct=0.1)])
memlog1 = learn.mem_stats.stats
learn = Learner(simple_dls(), simple_model(), loss_func=CrossEntropyLossFlat(), cbs=MemStatsCallback(label='Adam'), opt_func=Adam)
with learn.no_bar(): learn.fit(1, 1e-3, cbs=[ShortEpochCallback(pct=0.1)])
memlog2 = learn.mem_stats.stats
learn = Learner(simple_dls(), simple_model(), loss_func=CrossEntropyLossFlat(), cbs=MemStatsCallback(label='Adafactor'), opt_func=adafactor)
with learn.no_bar(): learn.fit(1, 1e-3, cbs=[ShortEpochCallback(pct=0.1)])
memlog3 = learn.mem_stats.stats
```

```python
plot_logs(memlog1, memlog2, memlog3)
```


![png](docs/images/output_12_0.png)

