# Epic-kitchen
## Github
[Official Pretrained Model](https://github.com/epic-kitchens/action-models)
* **Path on Srv**: `s1_md0/v-fanxu/junyidu/github/action-models`  

[TSM Backbone Code](https://github.com/mit-han-lab/temporal-shift-module)
* Borrows data pipeline

## Data
* **Raw Path**: `/s1_md0/v-fanxu/Extraction/youcook2/videos`
* **Extracted Img Path**: `/s1_md0/v-fanxu/Extraction/youcook2/imgs`
* **List Path**: `/s1_md0/v-fanxu/Extraction/youcook2/manifest.txt`
* **Epic Kitchen Verb List**: `/s1_md0/v-fanxu/junyidu/github/action-models/EPIC_verb_classes.csv` 

## Inference Result
* **Bulk**: `/s1_md0/v-fanxu/junyidu/github/action-models/predict_verb_bulk.txt`

## Setting
* [**Model download link**](https://data.bris.ac.uk/data/dataset/2tw6gdvmfj3f12papdy24flvmo)
* **Use**: [TSN, 8seg, Res50, RGB](https://data.bris.ac.uk/datasets/2tw6gdvmfj3f12papdy24flvmo/TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar)
* **Path on Srv**: `/s1_md0/v-fanxu/junyidu/download/TSN_arch=resnet50_modality=RGB_segments=8-3ecf904f.pth.tar`

## Preprocessing
Decode raw video
* In dataset's root folder
* `python vid2img.py videos imgs`

Generate manifest for video (bulk or clip)
* In the corresponding imgs folder
* `bash /s1_md0/v-fanxu/junyidu/github/action-models/youcook2/generate_manifest.sh`

Fetch framerate for each video
* In videos folder
* `bash /s1_md0/v-fanxu/junyidu/github/action-models/youcook2/fetch_framerate.sh`

## Other Action Dataset
* **Kinetics**
  * Actions: 400
  * Min Clip for an Action: ~400
  * Tot Clips: 306k
  * Length: 10s
  * Resolution: Variable (Usually normalized to 340px)
* **UCF101**
  * Actions: 101
  * Min Clip for an Action: ~100
  * Tot Clips: 13k
  * Mean Length: 7.2s
  * Resolution: 320x240

## Paper
* [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383)
  * Dataset


## Experiments
* **Official Demo**(Rand): `bash rundemo.sh`
* **Extract**: `bash runextract.sh`