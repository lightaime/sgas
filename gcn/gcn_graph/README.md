# SGAS on Biological Networks ([PPI Dataset](https://arxiv.org/abs/1707.04638)) 
### Search 
We search each model on one GTX 1080Ti.
Search with `SGAS Cri2`:   
``` 
python train_search.py --use_history --random_seed --data ../../data
```
Search with `SGAS Cri1`:   
``` 
python train_search.py --random_seed --data ../../data
```
Just need to set `--data` into your desired data folder, dataset will be downloaded automatically.

### Train
We train each model on one tesla V100.

For training the best architecture (`Cri1_PPI_10`, `Cri1_PPI_Best`) with 5 cells and 512 filters (the best large architecture), run:
``` 
python main_ppi.py --phase train --arch Cri1_PPI_Best --num_cells 5 --init_channels 512 --data ../../data/ppi
```
Set `--arch` to any architecture you want. (One can find more architectures from `genotyps.py`) 

### Test
#### Pretrained Models

Our pretrained models can be found from [Google Cloud](https://drive.google.com/drive/folders/1sjLfOpYUYyBSI14G8-vFScZPRaZCXart?usp=sharing).

Use the parameter `--model_path` to set a specific pretrained model to load. For example,  

test the best large architecture using `SGAS Cri2` (expected micro-F1: 99.46%): 
```
python main_ppi.py --phase test --arch Cri2_PPI_Best --num_cells 5 --init_channels 512 --model_path log/Cri2_PPI_Best_l5_c512.pt
```

test the best large architecture using `SGAS Cri1` (expected micro-F1: 99.46%): 
```
python main_ppi.py --phase test --arch Cri1_PPI_Best --num_cells 5 --init_channels 512 --model_path log/Cri1_PPI_Best_l5_c512.pt
```

test the best small architecture using `SGAS Cri1` (expected micro-F1: 98.89%): 
```
python main_ppi.py --phase test --arch Cri1_PPI_Best --num_cells 5 --init_channels 64 --model_path log/Cri1_PPI_Best_l5_c64.pt
```

