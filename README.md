Code for federated open-world semi-supervised contrastive learning.
To train on Pathmnist, run
'''
python main.py --dataset pathmnist --algorithm foscr --lbl_percent 10 --novel_percent 4 --batchsize 512 --E 4 --lr 0.1 
'''
