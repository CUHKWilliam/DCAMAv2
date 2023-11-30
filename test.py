r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset
from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset # FSDataset4SAM
# from transformers import SamProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import pickle
import pycocotools.coco as COCO




ref_kvs_path = "/research/d6/rshr/xjgao/twl/data/COCO2014/ref_image_cache_l_100.pkl"
print("a")
with open(ref_kvs_path, "rb") as f:
    ref_kvs = pickle.load(f)
print("b")
ks = []
vs = []
class_ids_s = []
for i in range(len(ref_kvs[0])):
    ks.append(ref_kvs[0][i])
    vs.append(ref_kvs[1][i])
    class_ids_s.append(ref_kvs[1][i]['class_id'])

ks = torch.stack(ks, dim=0)
class_ids_s = torch.tensor(class_ids_s).long()
_, class_ids_s = torch.unique(class_ids_s, return_inverse=True)
coco_test = COCO.COCO("/research/d6/rshr/xjgao/twl/data/COCO2014/instances_val2014.json")
imgs = coco_test.loadImgs(coco_test.getImgIds())
anns = coco_test.loadAnns(coco_test.getAnnIds())
img_names = []
for i in range(len(imgs)):
    img_names.append(imgs[i]['file_name'])
img_names = np.array(img_names)

# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

img_size = 384
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(img_mean, img_std)])

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
            # q_image = Image.fromarray((q_image).astype(np.uint8))
            # img_t = transform1(q_image)
            # with torch.no_grad():
            #     features_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0))
            # features = features_dict['x_norm_patchtokens']
            # q = features[0].mean(0).detach().cuda()
            # ks2 = ks[class_ids_s == class_ids[0]].cuda()
            selected_idx = torch.where(class_ids_s == class_ids[0])[0].tolist()
            vs2 = [vs[i] for i in selected_idx]
            import ipdb;ipdb.set_trace()
            score = (ks2 * q).sum(-1) / torch.norm(ks2, dim=-1) / torch.norm(q)
            # indices = torch.argsort(score, descending=True)[:nshot]
            indices = torch.arange(len(score)).long()[:nshot]
            score = score[indices]
            # indices = torch.cat([indices[0].unsqueeze(0), indices[torch.logical_and(score > 0.8, torch.logical_not(score == score[0]))]], dim=0)
            print(len(indices))
            support_imgs_ = []
            support_masks_ = []
            support_names_ = []
            for i, idx in enumerate(indices):
                v = vs2[idx.item()]
                # if i == 0:
                #     v['support_img'] = data['orig_query_img']
                #     v['support_mask'] = data['orig_query_mask'].detach().cpu().numpy()
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

            import cv2
            q_image.save("debug.png")
            cv2.imwrite("debug2.png", (q_mask.detach().cpu().numpy() * 255).astype(np.uint8))
            Image.fromarray(vs2[indices[0]]['support_img']).save("debug3.png", )
            cv2.imwrite("debug4.png", (vs2[indices[0]]['support_mask'] * 255).astype(np.uint8))
            cv2.imwrite("debug5.png", data['orig_query_img'])
            import ipdb; ipdb.set_trace()
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

def test(model, dataloader, nshot):
    r""" Test """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # batch = process_batch4SAM(batch)
        # 1. forward pass
        batch = utils.to_cuda(batch)

        ## TODO: substitute the first support image by the query image
        # batch['support_imgs'][:, 0, :, :, :] = batch['query_img']
        # batch['support_masks'][:, 0, :, :] = batch['query_mask']
        ## TODO: end

        nshot = batch['support_imgs'].size(1)
        pred_mask = model.module.predict_mask_nshot(batch, nshot=nshot)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  iou_b=area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    Logger.initialize(args, training=False)

    # Model initialization
    model = DCAMA(args.backbone, args.feature_extractor_path, args.use_original_imgsize)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())


    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    params = model.state_dict()
    state_dict = torch.load(args.load)

    for k1, k2 in zip(list(state_dict.keys()), params.keys()):
        state_dict[k2] = state_dict.pop(k1)
    model.load_state_dict(state_dict, strict=True)
    # TODO:
    # for i in range(len(model.model.DCAMA_blocks)):
    #     torch.nn.init.constant_(model.model.DCAMA_blocks[i].linears[1].weight, 0.)
    #     torch.nn.init.constant_(model.model.DCAMA_blocks[i].linears[1].bias, 1.)
    model = nn.DataParallel(model)
    model.to(device)
    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize, args.vispath)

    # Dataset initialization
    FSSDataset.initialize(img_size=384, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
