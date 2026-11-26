"""Scene-only routing with no head appearance, geometry or head priority."""

import torch

from gazelle.routing.router import CoverageAwareSpatialRouter, RoutingOutput


class UnconditionalSpatialRouter(CoverageAwareSpatialRouter):
    def forward(self, scene_features, bboxes):
        batch, _, height, width = scene_features.shape
        # Only query counts are used to expose one support per decoder query.
        # Box coordinates and head-region features never enter this computation.
        mapping = torch.tensor([i for i, boxes in enumerate(bboxes) for _ in boxes],
                               device=scene_features.device, dtype=torch.long)
        logits = self.score_head(torch.tanh(self.scene_projection(scene_features))).squeeze(1)
        probabilities = torch.softmax(logits.float().flatten(1) / self.temperature, dim=1).view_as(logits)
        count = min(height * width, max(1, round(self.keep_ratio * height * width)))
        scores = probabilities.flatten(1) + 2 * self._escape_mask(
            height, width, scene_features.device).flatten().to(probabilities.dtype)
        indices = scores.topk(count, dim=1, sorted=False).indices.sort(dim=1).values
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)
        return RoutingOutput(
            support_logits=logits.index_select(0, mapping),
            support_probs=probabilities.index_select(0, mapping),
            person_to_image=mapping,
            person_head_masks=scene_features.new_zeros((len(mapping), height, width)),
            image_union_probs=probabilities,
            image_hard_masks=mask.view(batch, height, width),
            image_keep_indices=indices,
            actual_keep_ratio=count / (height * width),
        )


def without_person_conditions(router):
    replacement = UnconditionalSpatialRouter(
        router.scene_projection.in_channels, router.scene_projection.out_channels,
        keep_ratio=router.keep_ratio, temperature=router.temperature,
        escape_tokens=router.escape_tokens)
    replacement.load_state_dict(router.state_dict())
    return replacement
