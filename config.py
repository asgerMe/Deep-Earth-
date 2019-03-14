training_runs = 1000000
param_state_size = 8
data_path = "D:\PhD"
output_dir = ''
grid_dir = ''
sb_blocks = 1
velocity_field_name = ''
encoder_mlp_layers = 1
n_filters = 128
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
sdf_state=2
start_integrator_training = 50000
save_gif = 2000

path_e = ''
path_i = ''
test_field_path = ''

train_integrator_network = True
f_integrator_network = 0
sequence_length = 30
save_freq = 5000

tensor_board = ''
meta_graphs = ''
gif_path = ''

encoder_name = 'encoder'
integrator_name = 'integrator'
batch_size = 5

conv = False

fem_loss = True