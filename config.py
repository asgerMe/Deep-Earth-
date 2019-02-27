training_runs = 1000000
param_state_size = 8
data_path = "D:\PhD\AI_Data"
output_dir = ''
grid_file = 'AI_Data0_30_10_grid_data.npy'
sb_blocks = 1
velocity_field_name = 'SIM_data.npy'
encoder_mlp_layers = 1
n_filters = 16
use_fem=False
data_size = 32
alt_dir = ''
inference = True
lr_min = 0.0000025
lr_max = 0.0001
lr_update = 'decay'
resample = False
period = 5000
f_tensorboard = 10

train_integrator_network = True
f_integrator_network = 0
sequence_length=30
save_freq = 5000

tensor_board = ''
meta_graphs = ''

encoder_name = 'encoder'
integrator_name = 'integrator'