# CityInsight

**Cityinsight** is an implementation of the paper CityInsight: Incorporating Diffusion Model-based Building Footprint Segmentation into Urban Vitality Analysis.

**Note**：This repository is undergoing revisions, and our paper is still under review.

## Installation

```
conda create --n cityinsight python=3.8 && conda activate cityinsight
pip install -r requirement.txt
```

## Dataset

### Download the dataset

For SpaceNet V1 & V2:

For WHU Building:

For Inria Aerial Image:

For Massachusetts:

### Data processing

The dataset folder under */dataset* should as follows.

```
data
|----WHU
|--------train
|--------val
|--------test
|----Inria
|----SV
|----SV2
|----Massachusetts
```

# Get start

For training, run `python scripts/train.py --data_name`

For testinng, run `python scripts/sample.py --data_ame`

## Thanks




