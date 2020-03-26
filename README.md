# Requirements

This project was done and tested using Python 3.7 so it should work for Python 3.7+.

To create a conda virtual environment that would work, run

```
  conda env create -f environment.yaml
```

# Usage

## Dataset

### Custom dataset

Custom dataset would need to be implemented by:
1. Creating a class deriving from ```torch.utils.data.Dataset```;
2. Modifying ```datasets/__init__.py``` to include your new class.

## Train

### Train

To train, run
```
  python train.py --config path/to/config/file [--gpus gpu_id] [--debug]
```

Arguments:
```
  --config: path to configuration file
  --gpus: gpu id to be used
  --debug: to save the weights or not
```

For example:
```
  python train.py --config configs/train/debug_ircad.yaml --gpus 0 --debug
```

### Config

Modify the default configuration file (YAML format) to suit your need, the properties' name should be self-explanatory.

### Result

All the result will be stored in the ```runs``` folder in separate subfolders, one for each run. The result consists of the log file for Tensorboard, the network pretrained models (best metrics, best loss, and the latest iteration).

#### Training graph

This project uses Tensorboard to plot training graph. To see it, run

```
  tensorboard --logdir=logs
```

and access using the announced port (default is 6006, e.g ```http://localhost:6006```).

#### Pretrained models

The ```.pth``` files contains a dictionary:

```
  {
      'epoch':                the epoch of the training where the weight is saved
      'model_state_dict':     model state dict (use model.load_state_dict to load)
      'optimizer_state_dict': optimizer state dict (use opt.load_state_dict to load)
      'log':                  full logs of that run
      'config':               full configuration of that run
  }
```

## Eval

To test a pretrained model, run
```
  python eval.py --weight path/to/pretrained/model [--gpus 0] [--vis] [--output visualization/dir]
```

Arguments:
```
  --weight: path to pretrained model
  --gpus: gpu id to be used
  --vis: whether to save visualization or not
  --output: the directory where the visualization will be stored
```

For example:
```
  python eval.py --weight runs/test_best_acc.pth --gpus 0 --vis --output vis/best_acc
```
