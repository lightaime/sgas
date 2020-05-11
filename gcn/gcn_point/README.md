# SGAS on point clouds 
## Point Clouds Classification on [ModelNet](https://modelnet.cs.princeton.edu/)
### Search 
We search each model on one GTX 1080Ti on ModelNet10 dataset.  
Search with `SGAS Cri1`:   
``` 
python train_search.py --random_seed --data ../../data/
```
Search with `SGAS Cri2`:   
``` 
python train_search.py --use_history --random_seed --data ../../data
```

Just need to set `--data` into your desired data folder, ModelNet10 dataset will be downloaded automatically.

### Train
We train each model on one tesla V100. 

For training the best architecture searched using `SGAS Cri2` with 9 cells, 128 filters and k nearest neighbors 20 (the best large architecture), run:
```
python main_modelnet.py --phase train --arch Cri2_ModelNet_Best --num_cells 9 --init_channels 128 --k 20 --save Cri2_modelnet40_best_l9_c128_k20
```
Just need to set `--data` into your data folder, dataset ModelNet40 will be downloaded automatically.  
Set `--arch` to any architecture you want. (One can find more architectures from `genotyps.py`) 

If you suffer from `Out of Memory (OOM)` issue, you can train a compact model by runing:
```
python main_modelnet.py --phase train --arch Cri2_ModelNet_Best --num_cells 3 --init_channels 128 --k 9 --save Cri2_modelnet40_best_l3_c128_k9
```


### Test

Our pretrained models can be found from [Google Cloud](https://drive.google.com/drive/folders/1sjLfOpYUYyBSI14G8-vFScZPRaZCXart?usp=sharing).

Use the parameter `--model_path` to set a specific pretrained model to load. For example,

test the best large architecture using `SGAS Cri2` (expected overall accuracy: 93.23%): 
```
python main_modelnet.py --phase test --arch Cri2_ModelNet_Best --num_cells 9 --init_channels 128 --k 20 --model_path log/Cri2_modelnet40_best_l9_c128_k20.pt
```

test the best large architecture using `SGAS Cri1` (expected overall accuracy: 92.87%): 
```
python main_modelnet.py --phase test --arch Cri1_ModelNet_Best --num_cells 9 --init_channels 128 --k 20 --model_path log/Cri1_modelnet40_best_l9_c128_k20.pt
```

test the best small architecture  using `SGAS Cri2` (expected overall accuracy: 93.07%): 
```
python main_modelnet.py --phase test --arch Cri2_ModelNet_Best --num_cells 3 --init_channels 128 --k 9 --model_path log/Cri2_modelnet40_best_l3_c128_k9.pt
```

