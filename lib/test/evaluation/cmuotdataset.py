import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class CMUOTDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.cmuot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name.split('@')[0], sequence_name.split('@')[1])

        anno_path = os.path.join(seq_path, 'gt.txt')

        ground_truth_rect = load_text(anno_path, delimiter=',', dtype=np.float64)

        occlusion_label_path = os.path.join(seq_path, 'occluded.txt')
        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(occlusion_label_path, delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = os.path.join(seq_path, 'outside.txt')
        out_of_view = load_text(out_of_view_label_path, delimiter=',', dtype=np.float64, backend='numpy')

        modal_label_path = os.path.join(seq_path, 'modal.txt')
        modal = load_text(modal_label_path, delimiter=',', dtype=np.float64, backend='numpy')
        modal_bool = modal.astype(bool)

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = os.path.join(seq_path, 'img')

        frames_list = [os.path.join(frames_path, f'{frame_number:06d}.jpg') for frame_number in
                       range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'cmuot', ground_truth_rect.reshape(-1, 4),
                        target_visible=target_visible, modal=modal_bool)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['190-230m@008', '190-230m@014', '190-230m@021', '190-230m@031', '190-230m@032', '190-230m@035',
                         '190-230m@036', '190-230m@039', '190-230m@044', '190-230m@046', '190-230m@053', '190-230m@054',
                         '190-230m@059', '190-230m@071', '50-60m@023', '50-60m@025', '50-60m@028', '50-60m@033',
                         '50-60m@043', '50-60m@049', '50-60m@050', '50-60m@061', '50-60m@072', 'crowd@003', 'crowd@005',
                         'crowd@006', 'crowd@008', 'crowd@010', 'crowd@012', 'crowd@013', 'crowd@015', 'crowd@024',
                         'fog@001', 'fog@003', 'fog@004', 'fog@014', 'fog@015', 'fog@017', 'fog@019', 'fog@024',
                         'fog@025', 'fog@035', 'fog@040', 'fog@046', 'fog@048', 'fog@049', 'fog@050', 'gate@024',
                         'gate@026', 'gate@027', 'gate@033', 'gate@034', 'gate@036', 'gate@045', 'gate@050', 'gate@051',
                         'gate@052', 'gate@056', 'gate@070', 'gate@074', 'gate@077', 'gate@078', 'gate@079', 'gate@090',
                         'gate@091', 'gate@109', 'gate@113', 'gate2@003', 'gate2@020', 'gate2@032', 'hard@005',
                         'hard@006', 'hard@007', 'hard@012', 'hard@021', 'hard@022', 'hard@023', 'hard@043', 'hard@044',
                         'hard@045', 'hard@046', 'hard@047', 'hard@048', 'hard@049', 'hard@053', 'hard@054', 'hard@056',
                         'hard@058', 'hard@059', 'hard@073', 'hard@080', 'hard@082', 'hard@087', 'hard@088', 'hard@091',
                         'other@019', 'other@023', 'other@042', 'other@050', 'other@052', 'other@053', 'other@062',
                         'other@066', 'other2@004', 'other2@005', 'other2@006', 'other2@016', 'other2@018',
                         'other2@031', 'other2@042', 'other2@043', 'other2@060', 'other2@064', 'other2@066',
                         'other2@070', 'other2@074', 'other2@075', 'other2@076', 'other2@077', 'other2@081',
                         'other2@102', 'other3@001', 'other3@012', 'other3@020', 'other3@021', 'other3@050',
                         'other3@067', 'other4@017', 'other4@035', 'other4@036', 'other4@038', 'other4@040',
                         'other4@045', 'school@004', 'school@013', 'school@018', 'school@019', 'school@025',
                         'school@026', 'school@028', 'school@029', 'school@030', 'school@031', 'school@032',
                         'school@042', 'school@044', 'school@045', 'school@048', 'school@055', 'school@058',
                         'school@082', 'school@084', 'school@085', 'school@086', 'school@088', 'school@095',
                         'school@101', 'school@104', 'school@107', 'school@108', 'school2@009', 'school2@011',
                         'school2@035', 'school2@040', 'school2@041', 'school2@042', 'school2@050', 'school2@067',
                         'school2@071', 'school2@077', 'school2@093', 'school2@103', 'add@003', 'add@004', 'add@005',
                         'add@006', 'add@007', 'add@010', 'add@012', 'add@013', 'add@017', 'add@018', 'add@019',
                         'add@020', 'add@023', 'add@026', 'add@030', 'add@031', 'add@032', 'add@033', 'add@034',
                         'add@036', 'add@038', 'add@039', 'add@041', 'add@042', 'add@051', 'add@052', 'add@053',
                         'add@056', 'add@058', 'add@060', 'add@062', 'add@069', 'add@074', 'add@079', 'add@080',
                         'add@081', 'add@082', 'add@084', 'add@088', 'add@090', 'add@099', 'add@100', 'add@104',
                         'add@107']
        # sequence_list = ['190-230m@008', 'crowd@006', 'crowd@008','school2@011']

        return sequence_list
