import os.path

from pycocotools.coco import COCO
import numpy as np
import pickle
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14 = dinov2_vitl14.cuda()
transform1 = transforms.Compose([
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size
                                transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])
transform2 = transforms.Compose([
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size
                                transforms.ToTensor(),
                                ])
patch_size = dinov2_vitl14.patch_size # patchsize=14

#520//14
patch_h = 520//patch_size
patch_w = 520//patch_size
feat_dim = 1024


root_path = "/research/d6/rshr/xjgao/twl/data/COCO2014"
img_path = root_path + "/train2014/"
anno_path = root_path + "/instances_train2014.json"
cache_path = root_path + "/ref_image_cache_l_1000.pkl"

coco = COCO(anno_path)
cats = coco.loadCats(coco.getCatIds())
kvs = []
ks = []
vs = []
np.random.seed(0)
for cat in cats:
    print("cat:", cat)
    cat_id = cat['id']
    annos_ = coco.loadAnns(coco.getAnnIds(catIds=[cat_id]))
    annos = []
    for anno in annos_:
        bbox = anno['bbox']
        if bbox[2] < 70 or bbox[3] < 70:
            continue
        annos.append(anno)
    annos = np.random.choice(annos, size=(min(1000, len(annos)),))
    for anno in tqdm(annos):
        img = coco.loadImgs([anno['image_id']])[0]
        im = np.asarray(Image.open(os.path.join(img_path, img["file_name"])))
        mask = coco.annToMask(anno)
        bbox = anno['bbox']
        if len(im.shape) == 2:
            im = np.repeat(np.expand_dims(im, axis=-1), repeats=3, axis=-1)
        im_orig = im.copy()
        mask_orig = mask.copy()
        im = Image.fromarray(im[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(bbox[0] + bbox[2]), :])
        mask = mask[int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(bbox[0] + bbox[2])]
        # mask2 = transform2(Image.fromarray(np.repeat(np.expand_dims(mask, axis=-1), repeats=3, axis=-1)))[0]
        img_t = transform1(im).cuda()
        with torch.no_grad():
            features_dict = dinov2_vitl14.forward_features(img_t.unsqueeze(0))
        features = features_dict['x_norm_patchtokens']
        k = features[0].mean(0).detach().cpu()
        ks.append(k)
        v = {
            "support_img": im_orig,
            "support_mask": mask_orig,
            "support_name": img["file_name"],
        }
        vs.append(v)

kvs = [ks, vs]
with open(cache_path, "wb") as f:
    pickle.dump(kvs, f)