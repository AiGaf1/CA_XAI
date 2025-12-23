#----Project-variables------
scenario = 'mobile'
project = "Keystroke-XAI"
seed = 40
#----Data-Variables------
train_val_division = 0.80
#----Loss-Variables------
batches_per_epoch_train = 256
batches_per_epoch_val = 16
samples_per_batch_train = 512
samples_per_batch_val = 512
#----Model-Variables------
N_PERIODS = 16
sequence_length = 128
embedding_size = 512
#----Optimization-Variables------
lr_scheduler_T_max = 2500
epochs = 5
