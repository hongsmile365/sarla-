from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.network_path = '/DATA/hongzhihong/mywork/sarla/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/DATA/hongzhihong/mywork/sarla'
    settings.result_plot_path = '/DATA/hongzhihong/mywork/sarla/output/test/result_plots'
    settings.results_path = '/DATA/hongzhihong/mywork/sarla/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/DATA/hongzhihong/mywork/sarla/output'
    settings.segmentation_path = '/DATA/hongzhihong/mywork/sarla/output/test/segmentation_results'
    settings.cmuot_path = '/DATA/hongzhihong/sot/backup_not/uavcross'
    settings.cmuot_aligned_path = '/DATA/hongzhihong/sot/backup/uavcross'

    return settings

