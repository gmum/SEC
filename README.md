<p align="center">
  <p align="center">
    <h1 align="center">
    Spectral Energy Centroid: a Metric for Improving Performance and Analyzing Spectral Bias in Implicit Neural Representations
    </h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://www.linkedin.com/in/tdadela/"><strong>Tomasz Dądela</strong></a>
    ·
    <a target="_blank" href="https://www.linkedin.com/in/adam-kania-r/"><strong>Adam Kania</strong></a>
    ·
    <a target="_blank" href="https://www.linkedin.com/in/maciej-rut-395bb026b/"><strong>Maciej Rut</strong></a>
    ·
    <a target="_blank" href="https://matinf.uj.edu.pl/pracownicy/wizytowka?person_id=Przemyslaw_Spurek"><strong>Przemysław Spurek</strong></a>
  </p>
<p align="center">

This repository provides the official implementation of **Spectral Energy Centroid (SEC)**, 
a robust metric for quantifying signal complexity and analyzing the spectral bias of Implicit Neural Representations (INRs).


## Setup
Setup environment:
```
conda env create --file environment.yaml
conda activate sec
```
Log into [Weights & Biases](https://docs.wandb.ai/quickstart):
```
wandb login
```

## Improving your model with SEC-Conf

The algorithm of SEC-Conf is relatively straightforward:
1. Calculate the Spectral Energy Centroid (SEC) of the target signal.
2. Select the reference target image where the SEC value is the most similar to your target.
3. Train the model using the best configuration associated with the reference.

Instructions for using SEC-Conf:
```
usage: find_optimal_config.py [-h] [--size {S,M,L}] [--architecture {siren,finer,relu,wire,all}] image_path

Evaluate image frequency and find the best network configuration.

positional arguments:
  image_path            Path to the input image.

options:
  -h, --help            show this help message and exit
  --size {S,M,L}        Network size (S=2, M=3, L=4 hidden layers). Default: M
  --architecture {siren,finer,relu,wire,all}
                        Network architecture. Default: all
```
Example:
> python find_optimal_config.py example_image.png  --size=M --architecture=wire
```
Calculating frequency for image: example_image.png...
Calculated Target Frequency (Center of Mass): 5
Found 21 configurations matching the closest frequency (Delta: 0).

==================================================
BEST CONFIGURATION (Highest Mean PSNR)
==================================================
Architecture    : wire
Hidden Layers   : 3
Hidden Features : 180
Sigma           : 1
Mean PSNR       : 38.6479
==================================================
python src/train.py --datasets='["example_image.png"]' --conf=configs/wire_base.yaml model.omega_0=1 model.hidden_layers=3 model.hidden_features=180
==================================================
```

## Experiments

#### Image experiments

To train a SEC-configured model use the command returned by `find_optimal_config.py`. 
In general, to train a model use the `src/train.py` script and pass appropriate parameters. 
E.g. for Siren (ω = 100):
```
python src/train.py --datasets='["Animal/pexels-photo-69350_1__13.png",]' \
  --conf=configs/siren_base.yaml model.omega_0=100 \
  logger.save_dir=...    checkpoints_path=...
```

To change which model is trained, select appropriate config from:
- `configs/siren_base.yaml`
- `configs/relu_base.yaml`
- `configs/finer_base.yaml`
- `configs/wire_base.yaml`


**Debugging:** By default, models are trained on GPU but debugging on CPU is also possible:
```
python src/train.py --datasets='["Animal/pexels-photo-69350_1__13.png"]' \
  --conf=configs/siren_base.yaml \
  epochs=200 logger.val_check_interval=20 \
  accelerator=cpu \
  logger.group=debug 
```

## Dataset
For our experiments, we used 100 images randomly selected from [LIU4K-v2 Dataset](https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html) (25 per category). You can find the specific list of selected images in `image_names.txt`.

## **Citation**

```
@misc{dadela2026sec,  
  title={Spectral Energy Centroid: a Metric for Improving Performance and Analyzing
Spectral Bias in Implicit Neural Representations},  
  author={Tomasz Dądela and Adam Kania and Maciej Rut and Przemysław Spurek},  
  booktitle = {arXiv},  
  year={2026}  
}
```
