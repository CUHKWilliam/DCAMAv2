python ./test.py --datapath "/research/d6/rshr/xjgao/twl/data" \
                 --benchmark coco \
                 --fold 3 \
                 --bsz 1 \
                 --nworker 1 \
                 --backbone swin \
                 --feature_extractor_path "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/backbones/swin_base_patch4_window12_384_22kto1k.pth" \
                 --logpath "./logs" \
                 --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/log/train/fold_3_ft_v0_swin/best_model.pt" \
                 --nshot 30

		 # --load "/research/d6/rshr/xjgao/twl/logistic_project/DCAMA/checkpoint/coco-20i/resnet50_fold0.pt" \
#                 --visualize
#checkpoint/coco-20i/resnet50_fold0.pt
# log/train/fold_0_ft_v0/best_model.pt
