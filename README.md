## About The Project

This is a food classifier built under the specification for the first challenge of the contest [Reto Enseña Oracle](https://nuwe.io/dev/competitions/reto-ensena-oracle-espana/clasificacion-imagenes-reto_1)

The main functionality of the project can be summarized as follows:

* Usage of predefined [ResNet152](https://arxiv.org/pdf/1512.03385.pdf) model.
* Transfer learning using ([IMAGENET1K_V2](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html#torchvision.models.ResNet152_Weights)) weights.
* Fine tune the model using [provided dataset](https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Oracle/oracle_CV.zip)
  * Segregate a portion of the dataset for validation purposes and hence avoiding overfitting.
  * Data augmentation using transforms.
* Inference and result saving for test dataset as _json_ file.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built Using

Base technologies:

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)

Additional dependencies:

* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Getting Started

Given that [python3](https://www.python.org/downloads/) and [pip](https://pypi.org/project/pip/) are installed and correctly configured in the system, and that you have a [CUDA-capable hardware](https://developer.nvidia.com/cuda-gpus) installed, you may follow these steps.

### Prerequisites

* [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) version 11.0 or above is correctly installed.
* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) version 7 or above is correctly installed.

### Installation

1. Clone this repository locally.

```bash
git clone git@github.com:sperezacuna/oracle-challenge-f1.git
```
2. Create python [virtual environment](https://docs.python.org/3/library/venv.html) and activate it (**recommended**)

```bash
python -m venv env
source env/bin/activate 
```

3. Install all required dependencies.

```bash
pip install -r requirements.txt
```

### Execution

To train a new model based on the [provided dataset](https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Oracle/oracle_CV.zip).

* Download the [dataset]((https://challenges-asset-files.s3.us-east-2.amazonaws.com/Events/Oracle/oracle_CV.zip)) and copy all content from _dataset/all\_imgs_ to _data/processed/all\_imgs_
* Execute:
  ```bash
  python src/model/generate-model.py
  ```
* The generated model (along with a graph of training statistics) will be saved at _models/resnet152_

If you do not want to train a new model, a model pretrained by us can be found [here](https://drive.google.com/drive/folders/1fYlo8V8_GKCog4U4gs-7nwwlejGfZOjI?usp=share_link)

To infer the output values for the provided _data/processed/test.csv_ dataset (**given that a trained model is placed at _models/resnet152_ and has proper naming**):

* Execute:
  ```bash
  python src/process/process-data.py
  ```
* The generated output _json_ will be saved at _results/f1-[MODEL-UUID]_

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributing

This project is being developed during the course of a competition, so PRs from people outside the competition will **not** be **allowed**. Feel free to fork this repo and follow up the development as you see fit.

Don't forget to give the project a star!

<p align="right">(<a href="#top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

Iago Barreiro Río - i.barreiro.rio@gmail.com

Santiago Pérez Acuña - santiago@perezacuna.com

Victor Figueroa Maceira - victorfigma@gmail.com

<p align="right">(<a href="#top">back to top</a>)</p>
