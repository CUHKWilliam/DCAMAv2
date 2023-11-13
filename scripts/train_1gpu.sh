python ./train_1gpu.py --datapath "/research/d6/rshr/xjgao/twl/data" \
           --benchmark coco \
           --fold 0 \
           --bsz 1 \
           --nworker 0 \
           --backbone resnet50 \
           --feature_extractor_path "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/backbones/resnet50_a1h-35c100f8.pth" \
           --logpath "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/log" \
           --lr 1e-3 \
           --nepoch 500 \
           --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
           --nshot 3