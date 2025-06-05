# traffic-sign-recognition

## Introduction

A traffic sign recognition focus on Taiwan, for more detail, please refer to

## Setup

```
conda env create -f environment.yml
conda activate traffic
pip install -r requirements.txt
```

## Generate Training Data

```
sh generate_and_organize.sh <bgs_path> <templates_path> <out_path> <train_images> <val_images>
```

## Train YOLO

```
python train_yolo.py
```

You can set some argument as follows

- `model`: YOLO model's name. Eefault "yolo11n.pt".
- `data_path`: Where label for YOLO are. Default "ImageSet/bigLabels".
- `yaml_path`: Where data information for YOLO are. Default "ImageSet/bigLabel_data.yaml".
- `data_path`: Where YOLO's ourput want to be, it will actually put under `runs/detect/`. Default "yolo".
- `testdata_path`: If you want to visualize performance of your yolo model please set your data path. If it is None, the test will not be run. Default None.
- `testdata_output`: Where your test output are. Note that if you didn't specify `testdata_path`, this argument will be useless. Default "yolo_test".

## Train NobileNet

```
python process_data.py
python train_mobile_net.py
```
Note that you should have your training data named "ImageSet" 

You can set some argument about train_mobile_net as follows

- `save_dir`: Where your MobileNet model want to be. Default "mobile_net".

## Predit Data

```
python train_yolo.py
```

We use Groq api for semantic understanding. **You should have your api key for groq first and save it to api.txt** Please register it at [https://console.groq.com/keys](https://console.groq.com/keys)
We provide 12 test image of Taiwan streetview from Google Street View inside `test_image`. You can also try your own test data

- `yolo_model_path`: Where your YOLO model are. Default "runs/detect/yolo/weights/best.pt".
- `mobilenet_model_path`: Where your MobileNet models' **directory** are. Default "mobile_net".
- `testdata_path`: Where your test Image are. Default "test_image".
- `yaml_small_path`: Where data information for small label(label for MobileNet) are. Default "ImageSet/smallLabel_data.yaml".
- `yaml_big_path`: Where data information for big label(label for YOLO) are. Default "ImageSet/bigLabel_data.yaml".
- `data_path`: Where YOLO's ourput want to be, it will actually put under `runs/detect/`. Default "yolo".
- `output_dir`: Where your test output are.