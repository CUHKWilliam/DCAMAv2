r""" Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation """
import copy
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

from .base.swin_transformer import SwinTransformer
from model.base.transformer import MultiHeadedAttention, PositionalEncoding
# from transformers import SamProcessor, SamModel, pipeline

class DCAMA(nn.Module):

    def __init__(self, backbone, pretrained_path, use_original_imgsize):
        super(DCAMA, self).__init__()

        self.backbone = backbone
        self.use_original_imgsize = use_original_imgsize

        # feature extractor initialization
        if backbone == 'resnet50':
            self.feature_extractor = resnet.resnet50()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 6, 3]
            self.feat_ids = list(range(0, 17))
        elif backbone == 'resnet101':
            self.feature_extractor = resnet.resnet101()
            self.feature_extractor.load_state_dict(torch.load(pretrained_path))
            self.feat_channels = [256, 512, 1024, 2048]
            self.nlayers = [3, 4, 23, 3]
            self.feat_ids = list(range(0, 34))
        elif backbone == 'swin':
            self.feature_extractor = SwinTransformer(img_size=384, patch_size=4, window_size=12, embed_dim=128,
                                            depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
            self.feature_extractor.load_state_dict(torch.load(pretrained_path)['model'])
            self.feat_channels = [128, 256, 512, 1024]
            self.nlayers = [2, 2, 18, 2]
        elif backbone == "SAM":
            self.feature_extractor = SamModel.from_pretrained("facebook/sam-vit-base", ignore_mismatched_sizes=True).cuda().vision_encoder
            self.feat_channels = [64, 64, 64, 64]
            self.nlayers = [3, 3, 3, 4]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.feature_extractor.eval()

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.backbone = backbone

        # self.SAM_feat_heads = nn.ModuleList(
        #     [nn.Conv2d(768, 64, 1, 1) for _ in range(13)]
        # )

    def forward(self, query_img, support_img, support_mask, training=True, nshot=1):
        n_support_feats = []
        with torch.no_grad():
            for k in range(nshot):
                support_feats_ = self.extract_feats(support_img[:, k])
                support_feats = copy.deepcopy(support_feats_)
                del support_feats_
                torch.cuda.empty_cache()
                if self.backbone == "SAM":
                    for i in range(13):
                        support_feats[i] = self.SAM_feat_heads[i](support_feats[i])
                    n_support_feats.append(support_feats)
                else:
                    n_support_feats.append(support_feats)
            query_feats_ = self.extract_feats(query_img)
            query_feats = copy.deepcopy(query_feats_)
            del query_feats_
            torch.cuda.empty_cache()
        if self.backbone == "SAM":
            for i in range(13):
                query_feats[i] = self.SAM_feat_heads[i](query_feats[i])
        logit_mask = self.model(query_feats, n_support_feats, support_mask.clone(), nshot=nshot, training=training)

        return logit_mask

    def extract_feats(self, img):
        r""" Extract input image features """
        feats = []

        if self.backbone == 'swin':
            _ = self.feature_extractor.forward_features(img)
            for feat in self.feature_extractor.feat_maps:
                bsz, hw, c = feat.size()
                h = int(hw ** 0.5)
                feat = feat.view(bsz, h, h, c).permute(0, 3, 1, 2).contiguous()
                feats.append(feat)

        elif self.backbone == 'resnet50' or self.backbone == 'resnet101':
            bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nlayers)))
            # Layer 0
            feat = self.feature_extractor.conv1.forward(img)
            feat = self.feature_extractor.bn1.forward(feat)
            feat = self.feature_extractor.relu.forward(feat)
            feat = self.feature_extractor.maxpool.forward(feat)

            # Layer 1-4
            for hid, (bid, lid) in enumerate(zip(bottleneck_ids, self.lids)):
                res = feat
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

                if bid == 0:
                    res = self.feature_extractor.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

                feat += res

                if hid + 1 in self.feat_ids:
                    feats.append(feat.clone())

                feat = self.feature_extractor.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        elif self.backbone == "SAM":
            feats2 = self.feature_extractor(img, output_hidden_states=True)[-1]
            feats = []
            for feat in feats2:
                feats.append(feat.permute(0, 3, 1, 2).contiguous())
        return feats

    def predict_mask_nshot(self, batch, nshot):
        r""" n-shot inference """
        query_img = batch['query_img']
        support_imgs = batch['support_imgs']
        support_masks = batch['support_masks']

        if nshot == 1:
            logit_mask = self(query_img, support_imgs, support_masks, training=False)
        else:
            with torch.no_grad():
                query_feats_ = self.extract_feats(query_img)
                query_feats = copy.deepcopy(query_feats_)
                del query_feats_
                torch.cuda.empty_cache()
                if self.backbone == "SAM":
                    for i in range(13):
                        query_feats[i] = self.SAM_feat_heads[i](query_feats[i])
                n_support_feats = []
                for k in range(nshot):
                    support_feats_ = self.extract_feats(support_imgs[:, k])
                    support_feats = copy.deepcopy(support_feats_)
                    del support_feats_
                    torch.cuda.empty_cache()
                    if self.backbone == "SAM":
                        for i in range(13):
                            support_feats[i] = self.SAM_feat_heads[i](support_feats[i])
                    n_support_feats.append(support_feats)

            ## TODO:
            MAX_SHOTS = 100
            if len(n_support_feats) > MAX_SHOTS:
                nshot = MAX_SHOTS
                n_simis = []
                for i in range(len(n_support_feats)):
                    n_support_feat = n_support_feats[i]
                    n_simi = []
                    for layer_idx in range(len(n_support_feat)):
                        n_support_feat_l = n_support_feat[layer_idx]
                        n_simi_l = []
                        for b in range(len(n_support_feat_l)):
                            n_support_feat_l_b = n_support_feat_l[b]
                            query_feat_l_b = query_feats[layer_idx][b]
                            support_mask_l_b = support_masks[b][i]
                            coords = torch.stack(torch.where(support_mask_l_b > 0), dim=-1)
                            ratio = query_feat_l_b.size(-2) / support_mask_l_b.size(-2)
                            coords = (coords * ratio).floor().long()
                            support_mask_l_b2 = torch.zeros((query_feat_l_b.size(-2), query_feat_l_b.size(-1))).bool().cuda()
                            support_mask_l_b2[coords[:, 1], coords[:, 0]] = True
                            support_mask_l_b = support_mask_l_b2
                            # import cv2
                            # support_mask_l_b = torch.from_numpy(cv2.resize(support_mask_l_b.detach().cpu().numpy(), dsize=(query_feat_l_b.size(-2), query_feat_l_b.size(-1)), interpolation=cv2.INTER_LINEAR)).bool().cuda()
                            # support_mask_l_b = (F.interpolate(support_mask_l_b.unsqueeze(0).unsqueeze(0), (query_feat_l_b.size(-2), query_feat_l_b.size(-1)), mode='bilinear', align_corners=True).squeeze(0).squeeze(0) != 0).bool()

                            # support_vec = n_support_feat_l_b[:, support_mask_l_b].mean(-1)
                            support_vec = n_support_feat_l_b.view(n_support_feat_l_b.size(0), -1).mean(-1)

                            # query_vec = query_feat_l_b.mean((-2, -1))
                            simi_inx = (support_vec @ query_feat_l_b.view(query_feat_l_b.size(0), -1)).argmax()
                            query_vec = query_feat_l_b.view(query_feat_l_b.size(0), -1)[:, simi_inx]
                            simi = (support_vec * query_vec).sum() / torch.norm(support_vec) / torch.norm(query_vec)
                            n_simi_l.append(simi)
                        n_simi_l = torch.stack(n_simi_l, dim=0)
                        n_simi.append(n_simi_l)
                    n_simi = torch.stack(n_simi, dim=0)
                    n_simis.append(n_simi)
                n_simis = torch.stack(n_simis, dim=0)

                # n_support_feats2 = [[] for _ in range(MAX_SHOTS)]
                # support_masks2 = []
                # for i in range(MAX_SHOTS):
                #     n_support_feats2[i] = [[] for i in range(len(n_support_feats[0]))]
                #     for layer_idx in range(len(n_support_feats[0])):
                #         n_support_feats2[i][layer_idx] = [None for _ in range(len(n_support_feats[0][0]))]
                # for layer_idx in range(len(n_support_feats[0])):
                #     for b in range(len(n_support_feats[0][0])):
                #         simi = n_simis[:, layer_idx, b]
                #         indices = torch.argsort(simi, descending=True)[:MAX_SHOTS]
                #         for i in range(len(indices)):
                #             n_support_feats2[i][layer_idx][b] = n_support_feats[indices[i]][layer_idx][b]
                # for i in range(MAX_SHOTS):
                #     for layer_idx in range(len(n_support_feats2[0])):
                #         n_support_feats2[i][layer_idx] = torch.stack(n_support_feats2[i][layer_idx], dim=0)
                # n_support_feats = n_support_feats2
                # support_masks = support_masks2


                # n_simis = n_simis.max(1)[0][:, 0]
                n_simis = n_simis.mean(1)[:, 0]

                support_masks = support_masks[:, n_simis.argsort(descending=True)[:MAX_SHOTS], :, :]
                n_support_feats = [n_support_feats[i] for i in n_simis.argsort(descending=True)[:MAX_SHOTS]]
            logit_mask = self.model(query_feats, n_support_feats, support_masks.clone(), nshot, training=False)

        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
        else:
            logit_mask = F.interpolate(logit_mask, support_imgs[0].size()[2:], mode='bilinear', align_corners=True)
        return logit_mask.argmax(dim=1)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        loss = self.cross_entropy_loss(logit_mask, gt_mask)
        loss = loss.mean()
        return loss

    def train_mode(self):
        self.train()
        self.feature_extractor.eval()


class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.mixer3 = nn.Sequential(nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))

        self.channel_compressor = nn.Sequential(

        )

    def forward(self, query_feats, support_feats, support_mask, nshot=1, training=True):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
            support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
            support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
            mask = torch.stack([F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True)
                                for k in support_mask])
            mask = mask.view(bsz, -1)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask, )
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask, )
            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask, )

            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))
        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
        coarse_masks1 = torch.stack(coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)

        coarse_masks1 = self.conv1(coarse_masks1)
        coarse_masks2 = self.conv2(coarse_masks2)
        coarse_masks3 = self.conv3(coarse_masks3)

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
        mix = coarse_masks1 + coarse_masks2
        mix = self.conv4(mix)

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
        mix = mix + coarse_masks3
        mix = self.conv5(mix)

        # skip connect 1/8 and 1/4 features (concatenation)
        support_feat = torch.stack([support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]).max(dim=0).values
        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)

        upsample_size = (mix.size(-1) * 2,) * 2
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)

        support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
        if query_feats[self.stack_ids[0] - 1].size(2) != mix.size(2):
            query_feats[self.stack_ids[0] - 1] = F.interpolate(query_feats[self.stack_ids[0] - 1], upsample_size, mode='bilinear', align_corners=True)
            support_feat = F.interpolate(support_feat, upsample_size, mode='bilinear', align_corners=True)
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)

        # mixer blocks forward
        out = self.mixer1(mix)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        out = self.mixer2(out)
        upsample_size = (out.size(-1) * 2,) * 2
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.mixer3(out)
        support_mask_size = (support_mask.size(-2), support_mask.size(-1))
        logit_mask = F.interpolate(logit_mask, size=support_mask_size)
        return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
        r""" bulid conv blocks """
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch,
                                                   kernel_size=ksz, stride=stride, padding=pad))
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)
