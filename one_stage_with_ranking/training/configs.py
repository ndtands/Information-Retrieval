import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = 'nguyenvulebinh/vi-mrc-base'
HYPERPARAMETER = {
    "accumulation_steps": 8,
    "batch_size": 16,
    "n_workers": 4,
    "max_length": 384,
    "epochs": 5,
    "n_fold": 5,
    "weight_decay": 1e-3,
    "lr": 3e-5 ,
    "is_schedule": True, 
    "is_scaler": False
}
PATH_DATA_TRAIN = "/media/Z/NDT/IR/dataset/train/train_stage1_ranking_17t12.csv"
PATH_WEIGHTED_F1 = "/media/Z/NDT/IR/one_stage_with_ranking/training/weight/best_f1.bin"
PATH_WEIGHTED_LOSS = "/media/Z/NDT/IR/one_stage_with_ranking/training/weight/best_loss.bin"