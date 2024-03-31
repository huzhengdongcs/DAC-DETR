# ------------------------------------------------------------------------
# DAC-DETR
# Copyright (c) 2023  University of Technology Sydney & Baidu Inc & Zhejiang University. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from .onetomany import Stage1Assigner, Stage2Assigner
from .dn_components import prepare_for_cdn,dn_post_process 
from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
    IA_BCE_loss,
)
from .deformable_transformer import build_deforamble_transformer
import copy
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, box_xyxy_to_cxcywh
from torchvision.ops.boxes import batched_nms

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(
            self,
            backbone,
            transformer,
            num_classes,
            num_feature_levels,
            aux_loss=True,
            with_box_refine=False,
            two_stage=False,
            num_queries_one2one=300,
            num_queries_one2many=0,
            mixed_selection=False,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one matching part
            num_queries_one2many: number of object queries for one-to-many matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        num_queries = num_queries_one2one
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection

        self.dn_number = 100
        self.dn_box_noise_scale = 0.4
        self.dn_label_noise_ratio =  0.5
        self.dn_labelbook_size = dn_labelbook_size = 91
        self.num_classes = 91
        self.hidden_dim = hidden_dim = transformer.d_model
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)


    def forward(self, samples: NestedTensor, targets: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0: self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        # normal_queries = 300

        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries, ]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one :, 0 : self.num_queries_one2one,] = True
        self_attn_mask[0 : self.num_queries_one2one, self.num_queries_one2one :,] = True

        if self.training:
            input_query_label, input_query_bbox, attn_mask, dn_meta, dn_meta_cdecoder =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,num_queries=self.num_queries,num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = dn_meta_cdecoder =  None
            attn_mask = ( torch.zeros([self.num_queries, self.num_queries, ]).bool().to(src.device) )


        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            hs_cdecoder,
            inter_references_cdecoder,
            anchors,
        ) = self.transformer(srcs, masks, pos, query_embeds, attn_mask, input_query_label, input_query_bbox)

        hs[0] += self.label_enc.weight[0,0]*0.0

        all_outs = []
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_cdecoder = []
        outputs_coords_cdecoder = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                if self.training and dn_meta['pad_size']>0:
                    init_reference =torch.cat([input_query_bbox, init_reference],dim=1).sigmoid()
                reference = init_reference
                reference_cdecoder = init_reference
            else:
                reference = inter_references[lvl - 1]
                reference_cdecoder = inter_references_cdecoder[lvl - 1]

            reference = inverse_sigmoid(reference)
            reference_cdecoder = inverse_sigmoid(reference_cdecoder)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            outputs_class_cdecoder = self.class_embed[lvl](hs_cdecoder[lvl])
            tmp_cdecoder = self.bbox_embed[lvl](hs_cdecoder[lvl])


            if reference.shape[-1] == 4:
                tmp += reference
                tmp_cdecoder += reference_cdecoder
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                tmp_cdecoder += reference_cdecoder

            outputs_coord = tmp.sigmoid()
            if self.training and  dn_meta['pad_size']>0:
                add_query = input_query_label.size(1)
            else:
                add_query = 0
            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one + add_query ])
            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one + add_query])
            outputs_coord_cdecoder = tmp_cdecoder.sigmoid()
            outputs_classes_cdecoder.append(outputs_class_cdecoder[:, 0: self.num_queries_one2one + add_query])
            outputs_coords_cdecoder.append(outputs_coord_cdecoder[:, 0: self.num_queries_one2one + add_query])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_cdecoder = torch.stack(outputs_classes_cdecoder)
        outputs_coords_cdecoder = torch.stack(outputs_coords_cdecoder)

        if self.training:
            dn_meta_one2one = dn_meta.copy()
            dn_meta_cdecoder_ = dn_meta.copy()
            if self.dn_number > 0 and dn_meta is not None:
                outputs_classes_one2one, outputs_coords_one2one, outs  = dn_post_process(outputs_classes_one2one, outputs_coords_one2one, dn_meta_one2one, self.aux_loss,self._set_aux_loss_dn, ca=False)
                if outs is not None:
                    dn_meta_one2one['output_known_lbs_bboxes'] = outs
                outputs_classes_cdecoder, outputs_coords_cdecoder, outs = dn_post_process(outputs_classes_cdecoder, outputs_coords_cdecoder, dn_meta_cdecoder_, self.aux_loss, self._set_aux_loss_dn_cdecoder, ca=True)
                if outs is not None:
                    dn_meta_cdecoder_['output_known_lbs_bboxes'] = outs
        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_cdecoder": outputs_classes_cdecoder[-1],
            "pred_boxes_cdecoder": outputs_coords_cdecoder[-1],
        }


        if self.training:
            out["dn_meta"] = dn_meta_one2one
            out["dn_meta_cdecoder"] = dn_meta_cdecoder_

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_classes_cdecoder, outputs_coords_cdecoder
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
                "pred_logits_cdecoder": enc_outputs_class,
                "pred_boxes_cdecoder": enc_outputs_coord,
                "anchors": anchors,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss_dn(self, outputs_class, outputs_coord, ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, }
            for a, b in
            zip(outputs_class[:-1], outputs_coord[:-1], )
        ]

    @torch.jit.unused
    def _set_aux_loss_dn_cdecoder(self, outputs_class, outputs_coord, ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits_cdecoder": a, "pred_boxes_cdecoder": b, }
            for a, b in
            zip(outputs_class[:-1], outputs_coord[:-1], )
        ]



    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_classes_cdecoder, outputs_coord_cdecoder):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_logits_cdecoder": c, "pred_boxes_cdecoder": d, }
            for a, b, c, d in
            zip(outputs_class[:-1], outputs_coord[:-1], outputs_classes_cdecoder[:-1], outputs_coord_cdecoder[:-1], )
        ]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        num_queries = 300
        self.one2many = Stage2Assigner(num_queries)
        self.stg1_assigner = Stage1Assigner()
        tau=1.5
        match_number=[1,1,1,1,1,1,1] 
        self.initialize_weight_table(match_number, tau)
        self.loc_weight = None

    def initialize_weight_table(self, match_number, tau):
        self.weight_table = torch.zeros(len(match_number), max(match_number))#.cuda()
        for layer, n in enumerate(match_number):
            self.weight_table[layer][:n] = torch.exp(-torch.arange(n) / tau)

    def loss_labels(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, log=True, one2one=False, enc=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        if one2one is False:
            assert "pred_logits_cdecoder" in outputs
            src_logits = outputs["pred_logits_cdecoder"]
            box_scores = outputs['pred_boxes_cdecoder']
        else:
            assert "pred_logits" in outputs
            src_logits = outputs["pred_logits"]
            box_scores = outputs['pred_boxes']
            log = False

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        target_classes_onehot_prob = target_classes_onehot.clone()

        f_cost_boxes = torch.ones_like(target_classes_onehot)
        target_boxes = torch.cat([t["boxes"][v[1]] for t,v in zip(targets, indices)], dim=0)

        if enc is False:
            if cdn is True:
                pos_idx = self._get_src_permutation_idx(indices)
                pos_idx_c = pos_idx + (target_classes_o.cpu(), )
                src_boxes = box_scores[pos_idx]
                loss_class,loc_weight = IA_BCE_loss(src_logits, pos_idx_c, src_boxes, 
                                    target_boxes, indices, num_boxes, 
                                    alpha=0.25,  gamma=2, 
                                    w_prime=1, cdn=cdn, one2one = one2one)
                self.loc_weight = loc_weight
            else:
                w_prime = self.weight_table[lay_nums].cuda()
                pos_idx = self._get_src_permutation_idx(indices)
                pos_idx_c = pos_idx + (target_classes_o.cpu(), )
                src_boxes = box_scores[pos_idx]
                loss_class,loc_weight = IA_BCE_loss(src_logits, pos_idx_c, src_boxes, 
                                    target_boxes, indices, num_boxes, 
                                    alpha=0.25,  gamma=2, 
                                    w_prime= w_prime, cdn=cdn, one2one = one2one)
                self.loc_weight = loc_weight
            losses = {"loss_ce": loss_class}
        else:
            loss_ce = (
                sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1] )
            losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False , enc=False ):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits_cdecoder"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, lay_nums, cost_boxes=None, cdn=False, enc=False, one2one=False):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        if one2one is False:
            assert "pred_boxes_cdecoder" in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs["pred_boxes_cdecoder"][idx]
        else:
            assert "pred_boxes" in outputs
            idx = self._get_src_permutation_idx(indices)
            src_boxes = outputs["pred_boxes"][idx]
            log = False

        if enc is False:
            loc_weight = self.loc_weight
        else:
            loc_weight = 1


        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {}
        losses['loss_bbox'] = (loc_weight* loss_bbox.sum(dim=-1)).sum() / num_boxes    
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses['loss_giou'] = (loc_weight * loss_giou).sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes,  lay_nums,  cost_boxes=None, cdn=False, enc=False,  **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes,  lay_nums,  cost_boxes=cost_boxes, cdn=cdn,   enc=enc, **kwargs)

    def forward(self, outputs, targets, balance_targets, g2=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        indices_onetone = self.matcher(outputs_without_aux, targets)
        if g2 is False:
            indices_onetomany, cost_boxes = self.one2many(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        num_boxes_balance = sum(len(t["labels"]) for t in balance_targets)
        num_boxes_balance = torch.as_tensor(
            [num_boxes_balance], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_balance)
        num_boxes_balance = torch.clamp(num_boxes_balance / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        losses = {}
        if self.training:
            dn_meta = outputs['dn_meta']
            dn_meta_cdecoder = outputs['dn_meta_cdecoder']


        if self.training and dn_meta_cdecoder and 'output_known_lbs_bboxes' in dn_meta_cdecoder:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta_cdecoder)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes_cdecoder = dn_meta_cdecoder['output_known_lbs_bboxes']
            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']

            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes_cdecoder, targets, dn_pos_idx, num_boxes*scalar,   lay_nums= len(outputs["aux_outputs"]) + 1 ,    cdn = True, enc=False, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
            losses_ = self.loss_labels(output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,  lay_nums= len(outputs["aux_outputs"]) + 1, cdn = True, one2one=True)
            losses_.update(self.loss_boxes(output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,  lay_nums= len(outputs["aux_outputs"]) + 1,  cdn = True, one2one=True))
            losses_ =  {k + f'_dn': v for k, v in losses_.items()}
            for key, value in losses_.items():
                if key + "_one2one" in losses.keys():
                    losses[key + "_one2one"] += value * 1
                else:
                    losses[key + "_one2one"] = value * 1

        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_bbox_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn_one2one'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)
        
        if g2 is False:
            for loss in self.losses:
                kwargs = {}
                losses.update(
                    self.get_loss(loss, outputs, targets, indices_onetomany, num_boxes_balance, lay_nums= len(outputs["aux_outputs"]) + 1 , cost_boxes = cost_boxes, cdn = False, enc=False, **kwargs)
                )
        losses_ = self.loss_labels(outputs, targets, indices_onetone, num_boxes,  lay_nums= len(outputs["aux_outputs"]) + 1, cost_boxes = cost_boxes, cdn = False, one2one=True)
        losses_.update(self.loss_boxes(outputs, targets, indices_onetone, num_boxes,  lay_nums= len(outputs["aux_outputs"]) + 1, cost_boxes = cost_boxes, cdn = False, one2one=True))

        for key, value in losses_.items():
            if key + "_one2one" in losses.keys():
                losses[key + "_one2one"] += value * 1
            else:
                losses[key + "_one2one"] = value * 1


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_onetone = self.matcher(aux_outputs, targets)
                if g2 is False:
                    indices_onetomany, cost_boxes  = self.one2many(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == "masks":
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == "labels":
                            # Logging is enabled only for the last layer
                            kwargs["log"] = False
                        l_dict = self.get_loss(
                            loss, aux_outputs, targets, indices_onetomany, num_boxes_balance, lay_nums= i + 1,  cost_boxes = cost_boxes,  cdn = False, enc=False, **kwargs
                        )
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

                losses_ = self.loss_labels(aux_outputs, targets, indices_onetone, num_boxes,  lay_nums= i + 1,  cost_boxes = cost_boxes,  cdn = False, one2one=True)
                losses_.update(self.loss_boxes(aux_outputs, targets, indices_onetone, num_boxes,  lay_nums= i + 1, cost_boxes = cost_boxes,  cdn = False, one2one=True))
                losses_ = {k + f"_{i}": v for k, v in losses_.items()}
                for key, value in losses_.items():
                    if key + "_one2one" in losses.keys():
                        losses[key + "_one2one"] += value * 1
                    else:
                        losses[key + "_one2one"] = value * 1


                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][i]
                    aux_outputs_known_cdecoder = output_known_lbs_bboxes_cdecoder['aux_outputs'][i]
                    l_dict={}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}
                        l_dict.update(self.get_loss(loss, aux_outputs_known_cdecoder, targets, dn_pos_idx, num_boxes*scalar,  lay_nums= i + 1, cdn = True,  enc=False,  **kwargs))
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    losses_ = self.loss_labels(aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,  lay_nums= i + 1, cdn = True,  one2one=True)
                    losses_.update(self.loss_boxes(aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,   lay_nums= i + 1, cdn = True, one2one=True))
                    losses_ = {k + f'_dn_{i}': v for k, v in losses_.items()}
                    for key, value in losses_.items():
                        if key + "_one2one" in losses.keys():
                            losses[key + "_one2one"] += value * 1
                        else:
                            losses[key + "_one2one"] = value * 1

                    
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    l_dict_ =  dict()
                    for key, value in l_dict.items():
                        if  key != 'cardinality_error_dn' + f'_{i}':
                            l_dict_[key + "_one2one"] = torch.as_tensor(0.).to('cuda')  
                    losses.update(l_dict)
                    losses.update(l_dict_)




        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets, onetone=False)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, num_boxes,  lay_nums=7, cdn=False, enc = True, **kwargs
                )
                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses
    
    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups
        return output_known_lbs_bboxes,single_pad,num_dn_groups




class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, topk=300):
        super().__init__()
        self.topk = topk
        print("topk for eval:", self.topk)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.topk, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results

class NMSPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = all_indexes // out_logits.shape[2]
        all_labels = all_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            pre_topk = score.topk(10000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.8)[:300]
            results.append({
                'scores': score[keep_inds],
                'labels': lbls[keep_inds],
                'boxes':  box[keep_inds],
            })

        return results



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != "coco" else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        mixed_selection=args.mixed_selection,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef

    # if args.use_dn:
    weight_dict['loss_ce_dn'] = args.cls_loss_coef
    weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
    weight_dict['loss_giou_dn'] = args.giou_loss_coef


    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        for key, value in weight_dict.items():
            if key == "loss_ce":
                aux_weight_dict[key + "_enc" ] = 2
            else:
                aux_weight_dict[key + "_enc"] = value
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2one"] = value
    weight_dict = new_dict


    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_g2"] = value
    weight_dict = new_dict


    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(
        num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha
    )
    criterion.to(device)
    postprocessors = {"bbox": NMSPostProcess()}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
