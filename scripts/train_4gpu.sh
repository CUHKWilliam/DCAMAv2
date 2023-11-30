python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=18032 \
./train.py --datapath "/research/d6/rshr/xjgao/twl/data" \
           --benchmark coco \
           --fold 0 \
           --bsz 1 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth" \
           --logpath "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/log" \
           --lr 1e-4 \
           --nepoch 50 \
           --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/swin_fold0.pt" \
           --nshot 2
#           --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
