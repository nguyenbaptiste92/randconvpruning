# Domain Generalization on Constrained Platforms: on the Compatibility with Pruning Techniques

Project to study the impact of neural network pruning in the Single Domain Generalization setting.

[Paper Accepted at Global IOT Summit 2022 (Dublin)](https://link.springer.com/chapter/10.1007/978-3-031-20936-9_20)

[Copy of the official repository](https://gitlab.emse.fr/b.nguyen/randconvpruning)

__This work benefited from the French Jean Zay supercomputer thanks to the AI dynamic access program.__

## How do I get set up? #
### Set Virtual Environment
 In order to use this project in the best condition, we recommend to use [uv](https://docs.astral.sh/uv/) to install the environment. To do so, go to the base folder of the alpine project (where the pyproject.toml file is) and execute the following command:

```sh
uv sync
```

Once created, you can activate the environment with the following command :

```sh
source .venv/bin/activate
```

To run a python file with [uv](https://docs.astral.sh/uv/), you can use the following command :
```sh
uv run train_digits.py 
```
## Single Domain Generalization algorithm

We used the method from the article [Robust and Generalizable Visual Representation Learning via Random Convolutions](https://openreview.net/pdf?id=BVSM0x3EDK6) as our Single domain Generalization method.

The code of their [official github repository](https://github.com/wildphoton/RandConv) is adapted in Tensorflow.

## Neural network pruning algorithms

The following pruning heuristics are supported:

* Magnitude heuristic ([Learning both weights and connections for efficient neural networks]([https://arxiv.org/pdf/1506.02626]), [Pruning filters for efficient convnets](https://arxiv.org/pdf/1608.08710))
* SNIP heuristic ([Snip: Single-shot network pruning based on connection sensitivity](https://arxiv.org/abs/1810.02340))
* Synflow heuristic ([Pruning neural networks without any data by iteratively conserving synaptic flow](https://arxiv.org/abs/2006.05467))
* FPGM heuristic ([Filter pruning via geometric median for deep convolutional neural networks acceleration](https://arxiv.org/abs/1811.00250))

## Folder organisation

Our project includes several folders:

* data: folder where the dataset are stored.
* src: folder with the python files.
* logs: folder where logs of the experiments are saved as tensorboard files.
* results: folder where logs of the experiments are saved as text files (Not mandatory : this is done with bash commands).

## Running experiments on Digits and RealWorld benchmarks

* `run_digits.sh` provided bash commands for an exemple on Digits benchmark. In this example, Resnet20 is trained on MNIST using random convolutions as data augmentation during 150 epochs. This network is pruned using Magnitude heuristic iteratively for several sparsity rates. After each pruning round, this network is re-trained using weights rewinding.

* `run_har.sh` provided bash commands for an exemple on RealWorld benchmark. In this example, a [temporal convolution neural network](https://dl.acm.org/doi/abs/10.1145/3380985) is trained on RealWorld (chest) using random convolutions as data augmentation during 70 epochs. This network is pruned using Synflow heuristic at initialization with a sparsity rate of 70%.

By modifying the settings (parser arguments) on thoses files, our experiments can be reproduced.

* You can select the gpu which will be used with the parameter "-g". To run the programm with several gpu ([tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)), use "--multi_gpu".

## Refactor

* Use CI/CD tools like [ClearML](https://clear.ml/) to save logs instead of tensorboard and local files

* Split Trainer into smaller function (rand_conv, pruning)

* Separate SNIP pruning from the rand conv

