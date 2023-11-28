from easydict import EasyDict as edict

env_settings = edict()

env_settings.ROOT = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0"
env_settings.DATA = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"
env_settings.EXPERIMENTS = ""
env_settings.CUDA_VISIBLE_DEVICES = 0
env_settings.MASTER_LIST = {
    "ones": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_ones.csv",
    "zeros" : "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv"
}
env_settings.CONFIG = "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/config.yaml"