## Introduction

This is the Pytorch implementation of our paper:`An Optimal Reverse Affine Maximizer Auction Mechanism for Task Allocation in Mobile Crowdsensing` .


## Requirements


* Python >= 3.7
* Pytorch 1.10.0
* Argparse
* Logging
* Tqdm
* Scipy

## Usage

### Generate the data

```bash
python generate_data.py
#You can set the number of users
```

### Train RAMANet

```bash
#poi=5
# user=2
python 2x5.py

# user=4
python 4x5.py

# user=6
python 6x5.py

# user=8
python 8x5.py

# user=10
python 10x5.py

# user=12
python 12x5.py
```

## Acknowledgement

Our code is built upon the implementation of <https://arxiv.org/abs/2305.12162>.

