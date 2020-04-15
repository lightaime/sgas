# Searching CNN architectures with SGAS
We run 10 independent searches to get 10 architectures with
`Criterion 1` or `Criterion 2` on on CIFAR-10. We evaluate the discovered architectures on CIFAR-10 and report the mean and
standard deviation of the test accuracy across those 10 models and the performance of the best model. 

We choose the 3 best performing cell architectures on CIFAR-10 for each Criterion and train them
on ImageNet. 

CIFAR-10 will be downloaded automatically. To obtain ImageNet dataset, please follow the instructions [here](https://github.com/pytorch/examples/tree/master/imagenet).

### Search on CIFAR-10
The search takes about 0.25 day (6 hours) on a single NVIDIA GTX 1080Ti. To search with `SGAS Cri2`, run:
``` 
python train_search.py --use_history
```
To search with `SGAS Cri1`, run:
``` 
python train_search.py
```

### Train on CIFAR-10
To train the best architectures (`Cri1_CIFAR_Best`, `Cri2_CIFAR_Best`) from scratch, run:
``` 
python train.py --auxiliary --cutout --arch Cri1_CIFAR_Best
```
or
``` 
python train.py --auxiliary --cutout --arch Cri2_CIFAR_Best
```
### Train on ImageNet
To train the best architectures (`Cri1_ImageNet_Best`, `Cri1_ImageNet_Best`) from scratch, run:
```
python train_imagenet.py --auxiliary --arch Cri1_ImageNet_Best --batch_size 1024 --learning_rate 0.5
```
or
```
python train_imagenet.py --auxiliary --arch Cri2_ImageNet_Best --batch_size 1024 --learning_rate 0.5
```
* We run these experiments on 8 Nvidia Tesla V100 GPUs for three days by setting `--batch_size 1024` and `--learning_rate 0.5`.

Set `--arch` to any architecture you want. (One can find more architectures from `genotyps.py`).

### Pretrained models

Our pretrained models can be found from [Google Cloud](https://drive.google.com/drive/folders/1sjLfOpYUYyBSI14G8-vFScZPRaZCXart?usp=sharing).

### Test pretrained models on CIFAR-10

Use the parameter `--model_path` to set a specific pretrained model to load. For example,
to test the best architecture `Cri1_CIFAR_Best` or `Cri2_CIFAR_Best`, run:
```
python test.py --auxiliary --arch Cri1_CIFAR_Best --model_path Cri1_CIFAR_Best.pt
```
* Expected result: 2.39% test error with 3.8M model params.

or 
```
python test.py --auxiliary --arch Cri2_CIFAR_Best --model_path Cri2_CIFAR_Best.pt
```
* Expected result: 2.44% test error with 4.1M model params.

### Test pretrained models on ImageNet
To test the best architecture `Cri1_ImageNet_Best` or `Cri2_ImageNet_Best`, run:
```
python test_imagenet.py --auxiliary --arch Cri1_ImageNet_Best --model_path Cri1_ImageNet_Best.pt
```
* Expected result:  24.2% top1 test error with 5.3M model params.

or
```
python test_imagenet.py --auxiliary --arch Cri2_ImageNet_Best --model_path Cri2_ImageNet_Best.pt
```
* Expected result: 24.1% top1 test error with 5.4M model params.
