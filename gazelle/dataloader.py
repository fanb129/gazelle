import torch
import json
import os
import copy
from PIL import Image
import numpy as np

import gazelle.utils as utils

def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data


def load_data_gazefollow(file):
    data = json.load(open(file, "r"))
    return data


class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        dataset_name,
        path,
        split,
        transform,
        in_frame_only=True,
        sample_rate=1,
        image_indices=None,
        augment=None,
        return_heatmap=None,
        records=None,
    ):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train" if augment is None else bool(augment)
        self.return_heatmap = (
            self.split == "train"
            if return_heatmap is None
            else bool(return_heatmap)
        )
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        
        if dataset_name == "gazefollow":
            self.data = (
                records
                if records is not None
                else load_data_gazefollow(
                    os.path.join(
                        self.path, "{}_preprocessed.json".format(split)
                    )
                )
            )
        elif dataset_name == "videoattentiontarget":
            if records is not None:
                raise ValueError("records is only supported for GazeFollow")
            self.data = load_data_vat(os.path.join(self.path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate)
        else:
            raise ValueError("Invalid dataset: {}".format(dataset_name))

        if image_indices is None:
            self.image_indices = tuple(range(len(self.data)))
        else:
            self.image_indices = tuple(int(index) for index in image_indices)
            if len(set(self.image_indices)) != len(self.image_indices):
                raise ValueError("image_indices must not contain duplicates")
            if any(
                index < 0 or index >= len(self.data)
                for index in self.image_indices
            ):
                raise IndexError("image_indices contains an out-of-range index")

        self.data_idxs = []
        for i in self.image_indices:
            for j in range(len(self.data[i]['heads'])):
                if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                    self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm = head_data['bbox_norm']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']


        img_path = os.path.join(self.path, img_data['path'])
        img = Image.open(img_path)
        img = img.convert("RGB")
        width, height = img.size

        if self.aug:
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']

            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            # update width and height and re-normalize
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]
        
        img = self.transform(img)
        
        if self.return_heatmap:
            heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64) # note for training set, there is only one annotation
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap
        else:
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width

    def __len__(self):
        return len(self.data_idxs)


class GazeFollowImageDataset(torch.utils.data.dataset.Dataset):
    """Return one image and all queried heads in that image for evaluation.

    ``GazeDataset`` intentionally preserves the original per-person training
    behavior. This image-grouped view exercises GazeLLE's actual multi-person
    contract: DINO runs once per image, the router unions all queried-person
    supports, and the decoder still emits one heatmap per person.
    """

    def __init__(
        self,
        path,
        split,
        transform,
        *,
        in_frame_only=True,
        image_indices=None,
        records=None,
    ):
        self.path = path
        self.split = split
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.data = (
            records
            if records is not None
            else load_data_gazefollow(
                os.path.join(self.path, "{}_preprocessed.json".format(split))
            )
        )

        if image_indices is None:
            candidate_indices = tuple(range(len(self.data)))
        else:
            candidate_indices = tuple(int(index) for index in image_indices)
            if len(set(candidate_indices)) != len(candidate_indices):
                raise ValueError("image_indices must not contain duplicates")
            if any(index < 0 or index >= len(self.data) for index in candidate_indices):
                raise IndexError("image_indices contains an out-of-range index")

        self.image_indices = []
        self.head_indices = {}
        for image_index in candidate_indices:
            selected_heads = [
                head_index
                for head_index, head in enumerate(self.data[image_index]["heads"])
                if not self.in_frame_only or head["inout"] == 1
            ]
            if selected_heads:
                self.image_indices.append(image_index)
                self.head_indices[image_index] = tuple(selected_heads)
        self.image_indices = tuple(self.image_indices)

    def __getitem__(self, idx):
        image_index = self.image_indices[idx]
        image_data = self.data[image_index]
        heads = [
            image_data["heads"][head_index]
            for head_index in self.head_indices[image_index]
        ]

        image = Image.open(os.path.join(self.path, image_data["path"])).convert("RGB")
        width, height = image.size
        image = self.transform(image)

        bboxes = [copy.deepcopy(head["bbox_norm"]) for head in heads]
        gazex = [copy.deepcopy(head["gazex_norm"]) for head in heads]
        gazey = [copy.deepcopy(head["gazey_norm"]) for head in heads]
        inout = torch.tensor([head["inout"] for head in heads])
        heights = [height] * len(heads)
        widths = [width] * len(heads)
        return image, bboxes, gazex, gazey, inout, heights, widths

    def __len__(self):
        return len(self.image_indices)

    @property
    def person_count(self):
        return sum(len(self.head_indices[index]) for index in self.image_indices)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


def collate_gazefollow_images(batch):
    images, bboxes, gazex, gazey, inout, heights, widths = zip(*batch)

    def flatten(groups):
        return [item for group in groups for item in group]

    return (
        torch.stack(images),
        list(bboxes),
        flatten(gazex),
        flatten(gazey),
        torch.cat(inout),
        flatten(heights),
        flatten(widths),
    )
