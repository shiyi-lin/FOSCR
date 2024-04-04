# FOSCR
This repo provides an implementation of FOSCR proposed in *Accelerating Early Detection of Emerging Disease via Federated Open-World Semi-Supervised Learning with Contrastive Representations*.
## Usage
To train on NCT-CRC dataset, run

```bash
python main.py --dataset nctcrc --algorithm foscr --lbl_percent 10 --novel_percent 4 --batchsize 512 --E 4 --lr 0.1 --arch RN18_simclr_CIFAR --w-supce 0.2 --w-supcon 0.2 --w-semicon 1 --proto_align True --w-proto 2 --seen_device 1
```
