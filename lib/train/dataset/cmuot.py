import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class CMUOT(BaseVideoDataset):
    """ CMUOT dataset.

    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().cmuot_dir if root is None else root
        super().__init__('CMUOT', root, image_loader)

        self.sequence_list = self._build_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))


    def _build_sequence_list(self, split=None):
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'cmuot_train_split.txt')
            elif split == 'test':
                file_path = os.path.join(ltr_path, 'data_specs', 'cmuot_test_split.txt')
            else:
                raise ValueError('Unknown split name.')
            # sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        else:
            raise ValueError('Set either split_name.')

        return sequence_list


    def get_name(self):
        return 'cmuot'

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "gt.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "occluded.txt")
        out_of_view_file = os.path.join(seq_path, "outside.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _read_modal(self, seq_path):
        modal_file = os.path.join(seq_path, 'modal.txt')
        with open(modal_file, 'r', newline='') as f:
            modal = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        return modal

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        modal = self._read_modal(seq_path)

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'modal': modal}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_sequence_path(self, seq_id):
        dir_seq = self.sequence_list[seq_id]
        dir, seq = dir_seq.split('@')
        return os.path.join(self.root, dir, seq)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_cross_modal(self):
        total = 0
        res = []
        for seq_id in range(self.get_num_sequences()):
            seq_path = self._get_sequence_path(seq_id)
            modal = self._read_modal(seq_path).tolist()
            bbox = self._read_bb_anno(seq_path)
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            visible = self._read_target_visible(seq_path) & valid.byte()
            for i, m in enumerate(modal):
                if modal[i - 1 if i != 0 else 0] != modal[i]:
                    if visible[i] == 1 and visible[i - 1 if i != 0 else 0] == 1:
                        res.append(str(seq_id) + '@' + str(i))
                        total += 1

        return res, total
