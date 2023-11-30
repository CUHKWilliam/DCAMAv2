r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset, FSDataset4SAM
# from transformers import SamProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F
# from torchvision import transforms
import pickle
import pycocotools.coco as COCO

import os
os.environ["NCCL_DEBUG"] = "INFO"

# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# transform1 = transforms.Compose([
#             transforms.Resize(520),
#             transforms.CenterCrop(518),  # should be multiple of model patch_size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=0.5, std=0.2)
#         ])
#
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
#
# ref_kvs_path = "/research/d6/rshr/xjgao/twl/data/COCO2014/ref_image_cache_l.pkl"
# with open(ref_kvs_path, "rb") as f:
#     ref_kvs = pickle.load(f)
#
# ks = []
# vs = []
# for i in range(len(ref_kvs[0])):
#     ks.append(ref_kvs[0][i])
#     vs.append(ref_kvs[1][i])
# ks = torch.stack(ks, dim=0)
# coco_test = COCO.COCO("/research/d6/rshr/xjgao/twl/data/COCO2014/instances_train2014.json")
# imgs = coco_test.loadImgs(coco_test.getImgIds())
# anns = coco_test.loadAnns(coco_test.getAnnIds())
# img_names = []
# for i in range(len(imgs)):
#     img_names.append(imgs[i]['file_name'])
# img_names = np.array(img_names)
#
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# img_size = 384
# img_mean = [0.485, 0.456, 0.406]
# img_std = [0.229, 0.224, 0.225]
# transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             transforms.Normalize(img_mean, img_std)])
def process_batch4SAM(batch_data, use_query_as_support=False):
    batch_size = len(batch_data)
    nshot = len(batch_data[0]['support_imgs'])
    query_images = []
    query_original_sizes = []
    query_reshaped_sizes = []
    query_masks = []
    query_names = []
    class_ids = []
    support_imgs = []
    support_masks = []
    support_names = []
    for b in range(batch_size):
        data = batch_data[b]
        query_images.append(data['query_img'])
        query_masks.append(data['query_mask'])
        # query_original_sizes.append(data['orig_query_imsize'])
        query_names.append(data['query_name'])
        class_ids.append(data['class_id'])

        rand_num = np.random.rand()
        if rand_num > 1.:
            ## TODO: use_query_as_support = True
            # if use_query_as_support:
            data['support_imgs'] = [data['query_img'], data['query_img'].clone()]
            data['support_masks'] = [data['query_mask'], data['query_mask'].clone()]
            data['support_names'] = [data['query_name'], data['query_name']]
            ## TODO: end
        elif rand_num > 0.:
            ## TODO: retrieve images by query as support
            q_image = data['orig_query_img']
            q_mask = data['orig_query_mask']
            q_name = data['query_name'].split("/")[-1]
            image_id = imgs[np.where(img_names == q_name)[0][0]]['id']
            anns = coco_test.loadAnns(coco_test.getAnnIds(imgIds=[image_id]))
            max_cnt = 0
            q_mask2 = q_mask
            for ann in anns:
                q_inst_mask = torch.from_numpy(coco_test.annToMask(ann)).float()
                cnt = torch.logical_and(q_inst_mask.bool(), q_mask.bool()).sum() / torch.logical_or(q_inst_mask.bool(),
                                                                                                    q_mask.bool()).sum()
                if cnt > max_cnt:
                    max_cnt = cnt
                    q_mask2 = q_inst_mask
            q_mask = q_mask2
            xys = torch.stack(torch.where(q_mask.bool())[::-1], dim=-1)
            xy_max, xy_min = xys.max(0)[0], xys.min(0)[0]
            # xy_max, xy_min = torch.min(xy_max + 50, torch.tensor([q_image.shape[1], q_image.shape[0]]).long()), torch.max(xy_min - 50, torch.tensor([0, 0]).long())
            q_image = Image.fromarray((q_image[xy_min[1]: xy_max[1], xy_min[0]: xy_max[0], :]).astype(np.uint8))
            img_t = transform1(q_image)
            with torch.no_grad():
                features_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0))
            features = features_dict['x_norm_patchtokens']
            q = features[0].mean(0).detach()
            score = (ks * q).sum(-1) / torch.norm(ks, dim=-1) / torch.norm(q)
            indices = torch.argsort(score, descending=True)[:nshot]
            score = score[indices]
            indices = torch.cat([indices[0].unsqueeze(0), indices[torch.logical_and(score > 0.7, torch.logical_not(score == score[0]))]], dim=0)
            print(len(indices))
            support_imgs_ = []
            support_masks_ = []
            support_names_ = []
            for idx in indices:
                v = vs[idx.item()]
                try:
                    support_img = transform(Image.fromarray(v['support_img']))
                except:
                    support_img = transform(v['support_img'])
                support_imgs_.append(support_img)
                support_mask = F.interpolate(torch.from_numpy(v['support_mask']).unsqueeze(0).unsqueeze(0).float(),
                                             support_img.size()[-2:], mode='nearest').squeeze()
                support_masks_.append(support_mask)
                support_names_.append(v['support_name'])

            data['support_names'] = support_names_
            data['support_imgs'] = support_imgs_
            data['support_masks'] = support_masks_

            # import cv2
            # q_image.save("debug.png")
            # cv2.imwrite("debug2.png", (q_mask.detach().cpu().numpy() * 255).astype(np.uint8))
            # Image.fromarray(vs[indices[0]]['support_img']).save("debug3.png", )
            # cv2.imwrite("debug4.png", (vs[indices[0]]['support_mask'] * 255).astype(np.uint8))
            # cv2.imwrite("debug5.png", data['orig_query_img'])
            # import ipdb; ipdb.set_trace()
            # # TODO: end
        support_images = data['support_imgs']
        support_msks = data['support_masks']
        support_ns = data['support_names']
        support_images2 = support_images
        support_masks2 = support_msks
        support_names2 = support_ns
        support_masks2 = torch.stack(support_masks2)
        support_images2 = torch.stack(support_images2, dim=0)
        support_imgs.append(support_images2)
        support_masks.append(support_masks2)
        support_names.append(support_names2)
    query_images = torch.stack(query_images, dim=0)
    query_masks = torch.stack(query_masks, dim=0)
    batch_data2 = {}
    batch_data2["query_img"] = query_images
    batch_data2["org_query_imsize"] = query_original_sizes
    batch_data2['query_name'] = query_names
    class_ids = torch.tensor(class_ids).long()
    batch_data2['class_id'] = class_ids
    batch_data2['query_mask'] = query_masks
    support_imgs = torch.stack(support_imgs, dim=0)
    support_masks = torch.stack(support_masks, dim=0)
    batch_data2['support_imgs'] = support_imgs
    batch_data2['support_masks'] = support_masks
    batch_data2['support_names'] = support_names
    return batch_data2


def train(epoch, model, dataloader, optimizer, training, shot=1):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # batch = process_batch4SAM(batch)
        # 1. forward pass
        batch = utils.to_cuda(batch)
        shot = batch['support_imgs'].size(1)

        logit_mask = model(batch['query_img'], batch['support_imgs'], batch['support_masks'], nshot=shot)
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, False)
    device = torch.device("cuda", args.local_rank)
    model.to(device)
    params = model.state_dict()
    state_dict = torch.load(args.load)
    params2 = params.copy()
    for key in params.keys():
        if "norm_base" in key:
            params2.pop(key)
    params = params2
    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)

    model.load_state_dict(state_dict, strict=False)

    ## TODO:
    for i in range(len(model.model.DCAMA_blocks)):
        torch.nn.init.constant_(model.model.DCAMA_blocks[i].linears[1].weight, 0.)
        torch.nn.init.constant_(model.model.DCAMA_blocks[i].linears[1].bias, 1.)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    if torch.distributed.get_rank() == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=args.nshot)
    if args.local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=args.nshot)

    # Train
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    start_epoch = 0

    for epoch in range(start_epoch, args.nepoch):
        dataloader_trn.sampler.set_epoch(epoch)
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True, shot=args.nshot)

        # evaluation
        if torch.distributed.get_rank() == 0:
            # with torch.no_grad():
            #     val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

            # Save the best model

            # if val_miou > best_val_miou:
            #     best_val_miou = val_mious
            Logger.save_model_miou(model, epoch, 1.)

            # Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            # Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            # Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            # Logger.tbd_writer.flush()

    if torch.distributed.get_rank() == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')
