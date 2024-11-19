import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from functools import reduce


class Segment:
    def __init__(self, image_idx, image_path, mask_ids, mask_feature, mask_score, mask_center, uuid=-1):
        self.image_idx = {image_idx}  # [1,]
        self.image_path = {image_path}  # [filename1,]
        self.init_idx = image_idx

        self.mask_center = torch.tensor([mask_center])  # [0.85,]
        self.mask_score = [mask_score]  # [0.85,]
        self.mask_feature = mask_feature  # [1, D] or [D]
        self.mask_ids = mask_ids  # [1, N] or [N]
        self.mask_size_list = [torch.sum(self.mask_ids)]  # [856,]

        self.merge_num = torch.tensor([1])
        self.instance_pt_count = torch.zeros_like(mask_ids, dtype=torch.int)  # [1, N] or [N]
        self.instance_pt_count[mask_ids] = 1
        self.contained = None
        self.containing = None

        self.uuid = uuid
        self.uuid_set = {uuid}

    def myadd(self, other):
        # image index and color path, union
        self.uuid_set = self.uuid_set.union(other.uuid_set)
        self.image_idx = self.image_idx.union(other.image_idx)
        self.image_path = self.image_path.union(other.image_path)
        self.init_idx = min(self.init_idx, other.init_idx)

        # segment score and mask size, append
        self.mask_center = self.mask_center + other.mask_center
        self.mask_score = self.mask_score + other.mask_score

        # segment feature, average
        # self.mask_feature = self.mask_feature * self.merge_num / (self.merge_num + 1) + other.mask_feature / (self.merge_num + 1)
        self.mask_feature = self.mask_feature + other.mask_feature
        self.merge_num = self.merge_num + other.merge_num

        # segment ids, logical or,
        self.mask_ids = torch.logical_or(self.mask_ids, other.mask_ids)
        #
        self.instance_pt_count = self.instance_pt_count + other.mask_ids.int()
        self.mask_size_list.append(torch.sum(other.mask_ids))

        return self

    def __add__(self, other):
        # image index and color path, union
        self.uuid_set = self.uuid_set.union(other.uuid_set)
        self.image_idx = self.image_idx.union(other.image_idx)
        self.image_path = self.image_path.union(other.image_path)
        self.init_idx = min(self.init_idx, other.init_idx)

        # segment score and mask size, append
        self.mask_center = self.mask_center + other.mask_center
        self.mask_score = self.mask_score + other.mask_score

        # segment feature, average
        # self.mask_feature = self.mask_feature * self.merge_num / (self.merge_num + 1) + other.mask_feature / (self.merge_num + 1)
        self.mask_feature = self.mask_feature + other.mask_feature
        self.merge_num = self.merge_num + other.merge_num

        # segment ids, logical or,
        self.mask_ids = torch.logical_or(self.mask_ids, other.mask_ids)
        self.instance_pt_count = self.instance_pt_count + other.instance_pt_count
        self.mask_size_list = self.mask_size_list + other.mask_size_list

        return self


class InstanceList(list):
    def get_values(self, key, idx: int = None):
        if idx is None:
            return [getattr(detection, key) for detection in self]
        else:
            return [getattr(detection, key)[idx] for detection in self]

    def get_stacked_values_torch(self, key, idx: int = None):
        values = []

        for detection in self:
            v = getattr(detection, key)
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)

            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx: int = None):
        values = self.get_stacked_values_torch(key, idx)
        return values.cpu().numpy()

    def slice_by_indices(self, index):
        """
        Return a sublist of the current list by indexing
        """
        new_self = type(self)()

        for i in index:
            if i < len(self):
                new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask):
        """
        Return a sublist of the current list by masking
        """
        new_self = type(self)()

        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self

    def delete_index(self, index):
        """
        Return a new instance list that doesn't contain the index
        """
        instance = type(self)()

        for idx, segment in enumerate(self):
            if idx not in index:
                instance.append(segment)

        return instance

    def delete_uuid(self, uuid):
        """
        Return a new instance list that doesn't contain the index
        """
        instance = type(self)()

        for idx, segment in enumerate(self):
            if segment.uuid not in uuid:
                instance.append(segment)

        return instance

    def pop_and_aggregate(self, index, pop=True):
        """
        Merge parts of the current segments into one, and return merged segment;
        Args:
            pop: whether to pop the partial segments
            index: index of the partial segments

        Returns:
            merged segment
        """
        merged_segment = reduce(lambda a, b: a + b, [self[i] for i in index])
        return merged_segment
