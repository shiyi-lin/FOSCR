# FOSCR
This repo provides an implementation of FOSCR proposed in *From Signals to Surveillance: Leveraging Federated AI for Rapid Detection of Emerging Health Threats*.
## Usage
To train on ColorectalSlides dataset, run

```bash
python main.py --dataset nctcrc --algorithm foscr --lbl_percent 10 --novel_percent 4 --batchsize 512 --E 4 --lr 0.1 --arch RN18_simclr_CIFAR --w-supce 0.2 --w-supcon 0.2 --w-semicon 1 --proto_align True --w-proto 2 --seen_device 1
```
