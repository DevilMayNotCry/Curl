# Requirements

```
Pytorch0.3.1 is needed. Very important!

Python >= 3.5 (my version is Python3.6)

imageio==2.9.0
numpy == 1.19.1
opencv-python==4.4.0.42
Pillow==7.2.0
scikit-image==0.17.2
scipy==1.5.2
torch==0.3.1
torchvision==0.2.2
tqdm==4.19.9
```



# changes

1. During training and validation、testing phase, images loaded are always **only one**.
2. No resized operations.
3. All images are resized to times of 8.
4. Some parameters of the network is changed.





# data

Tiny MIT-FiveK datasets. Listed below are some links for downloading it:

BaiduNetDisk: https://pan.baidu.com/s/1_VwqWqpPGw5piLxg8pWEZg    code: 0212

Google Drive: 



# infer

Pretrained models are avaliable at ./checkpoints/log2020-09-11_15-06-47/, with 30 epoches and valid PSNR of 25.09db on MIT-FiveK datasets.

During training, 1-4500 of expertC is used for training and 4501-50000 is used for testing.

Besides, all images for all process are resizd to max edge 512px. Therefore, results above are suitable for low resolution images. Although the model also works well upon high resolution images, the objective results may decrease a little.

Images to enhance are listed in ./datasets/test/input/

```shell
python CURL.py --regen
```

or more concret one version

```shell
python CURL.py --regen  -i  ./datasets/test/input/ -o ./results/ -c ./checkpoints/log2020-09-11_15-06-47/curl_validpsnr_25.092590592669_validloss_0.022607844322919846_epoch_30_model.pt
```

and the enhanced images are listed in ./results/.

![](./images/Snipaste_2020-09-12_16-03-34.jpg)



# test

input images:  ./datasets/test/input/

label images:  ./datasets/test/expertC_gt/

```shell
python CURL.py --test
```

or 

```shell
python CURL.py --test -i  ./datasets/test/input/ -l ./datasets/test/expertC_gt/ -c ./checkpoints/log2020-09-11_15-06-47/curl_validpsnr_25.092590592669_validloss_0.022607844322919846_epoch_30_model.pt
```

Listed below are some results:

![](./images/Snipaste_2020-09-12_16-04-53.jpg)



# train

```shell
python CURL.py --train
```

or

```shell
python CURL.py \
--train \
--train_dir ./datasets/train/ \
--train_images_list ./datasets/train_images_list.txt \
--valid_dir ./datasets/test/ \
--valid_images_list ./datasets/test_images_list.txt \
--num_epoch 30 \
--valid_every 1
```

and the trained models are save at /checkpoint/log_....../