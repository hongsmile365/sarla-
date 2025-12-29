import os.path


class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/DATA/hongzhihong/mywork/sarla'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/DATA/hongzhihong/mywork/sarla/tensorboard'    # Directory for tensorboard files.
        self.cmuot_dir = '/DATA/hongzhihong/sot/backup_not/uavcross'
        self.cmuot_aligned_dir = '/DATA/hongzhihong/sot/backup/uavcross'
        self.rgbnir_dir = '/DATA/hongzhihong/sot/RGBN/RGBNIR'
