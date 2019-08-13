# On the Convergence of AdaBound and its Connection to SGD

Repository of the technical report "On the Convergence of AdaBound and its Connection to SGD" [[PDF](https://github.com/lolemacs/adabound-and-csgdm/raw/master/paper.pdf)], including the implementation for the proposed bias-corrected dampened form of momentum SGD (CSGD for short).

[Pedro Savarese](https://ttic.uchicago.edu/~savarese)

CSGD is offered as a stand-alone PyTorch module in csgd.py.

## Requirements
```
PyTorch == 1.1.0
```
The code should also work earlier versions of PyTorch (e.g. 0.4.0).

## Using CSGD

CSGD can be used like any of the PyTorch built-in optimizers, for example:

```
import CSGD from csgd
optimizer = CSGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
```

Note that CSGD does not have 'nesterov' nor 'dampening' arguments (it uses standard heavy-ball momentum with dampening=momentum).

## Citation
An arXiv submission will be available soon. For now, you can cite this report as:

```
@misc{savarese2019,
  author = {Pedro Savarese},
  title = {On the Convergence of AdaBound and its Connection to SGD},
  year = {2019},
  howpublished = {\url{https://github.com/lolemacs/adabound-and-csgdm}}
}
```
