r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
        # ## TODO:
        # np.random.seed(0)
        # self.img_metadata = np.random.choice(self.img_metadata, len(self.img_metadata) // 10)

    def __len__(self):
        return 500 if self.split == 'trn' else 1000
        # return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
        orig_query_img = np.asarray(query_img).copy()
        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        orig_query_mask = query_mask.clone()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        
        support_imgs = [self.transform(support_img) for support_img in support_imgs]
        flag_all_same_shape = True
        for i in range(len(support_imgs)):
            if support_imgs[i].size()[1:] != support_imgs[0].size()[1:]:
                flag_all_same_shape = False
        if flag_all_same_shape:
            support_imgs = torch.stack(support_imgs, dim=0)
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs[midx].size()[-2:], mode='nearest').squeeze()
        if flag_all_same_shape:
            support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'orig_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample),
                 
                 'orig_query_img': orig_query_img,
                 'orig_query_mask': orig_query_mask,
                 }

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)

        ## TODO:
        # np.random.seed(0)
        # img_metadata_classwise2 = {}
        # for class_id in img_metadata_classwise.keys():
        #     img_metadata = img_metadata_classwise[class_id]
        #     img_metadata2 = np.random.choice(img_metadata, len(img_metadata) // 10)
        #     img_metadata_classwise2[class_id] = img_metadata2
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize


from tqdm import tqdm
class DatasetCOCOWithRef(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        img_metadata_classwise_tmp, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.ref_kvs = self.build_ref_bank(img_metadata_classwise_tmp, )
        self.img_metadata = self.build_img_metadata()
        ## TODO:
        # np.random.seed(0)
        # self.img_metadata = np.random.choice(self.img_metadata, len(self.img_metadata) // 10)

    def build_ref_bank(self, img_metadata_classwise):
        cache_path = os.path.join(self.base_path, "cache_ref_kvs_{}_{}.pkl".format(self.split_coco, self.nfolds))
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                ref_kvs = pickle.load(f)
        else:
            names = []
            for k in img_metadata_classwise.keys():
                names += img_metadata_classwise[k]
            ref_kvs = []
            for i in tqdm(range(len(names))):
                name = names[i]
                img = np.asarray(Image.open(os.path.join(self.base_path, name)).convert('RGB'))
                mask = self.read_mask(name).detach().cpu().numpy()
                for sem_id in np.unique(mask):
                    if sem_id == 0:
                        continue
                    msk = mask == sem_id
                    xy = np.stack(np.where(msk.astype(np.bool))[::-1], axis=-1)
                    xy_max, xy_min = np.max(xy, axis=0), np.min(xy, axis=0)
                    img2 = img[xy_min[1]: xy_max[1], xy_min[0]: xy_max[0], :].copy()
                    import ipdb; ipdb.set_trace()
        return ref_kvs



    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = [self.transform(support_img) for support_img in support_imgs]
        flag_all_same_shape = True
        for i in range(len(support_imgs)):
            if support_imgs[i].size()[1:] != support_imgs[0].size()[1:]:
                flag_all_same_shape = False
        if flag_all_same_shape:
            support_imgs = torch.stack(support_imgs, dim=0)
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs[midx].size()[-2:], mode='nearest').squeeze()
        if flag_all_same_shape:
            support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'orig_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)

        ## TODO:
        # np.random.seed(0)
        # img_metadata_classwise2 = {}
        # for class_id in img_metadata_classwise.keys():
        #     img_metadata = img_metadata_classwise[class_id]
        #     img_metadata2 = np.random.choice(img_metadata, len(img_metadata) // 10)
        #     img_metadata_classwise2[class_id] = img_metadata2
        return img_metadata_classwise , img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        np.random.seed(0)
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
