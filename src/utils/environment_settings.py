from easydict import EasyDict as edict

env_settings = edict()

env_settings.ROOT = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0"
env_settings.DATA = "/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/files"
env_settings.EXPERIMENTS = "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/experiments/"
env_settings.CUDA_VISIBLE_DEVICES = 0
env_settings.MASTER_LIST = {
    "ones": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_ones.csv",
    "zeros" : "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/master_df_zeros.csv"
}
env_settings.OCCURENCE_PROBABILITIES = {
    "ones": "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_ones.json",
    "zeros" : "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/data/disease_combination_distribution_ones.json"
}
env_settings.CONFIG = "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/config.yaml"
env_settings.PRETRAINED_PATH = {
    'ARK' : "/home/guests/usr_mlmi/arda/Unsupervised-Structured-Reporting-via-Cycle-Consistency/src/pretrained_models"
}
env_settings.TENSORBOARD_PORT = 1881