import argparse
import config
import train
import os
import util

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to training fields / See ... for Houdini based data generator")
parser.add_argument("-od", "--output_dir", default='', help="path to training fields / See ... for Houdini based data generator")

parser.add_argument("-mg", "--meta graph_dir", default='', help="graph saving directory")
parser.add_argument("-t", "--train", help="train the network", action='store_true')
parser.add_argument("-lss", "--latent_state_size", type=int, help= "size of the latent state space. Should be 2^n", default=8, choices=[8, 16, 32, 64, 128])
parser.add_argument("-sb", "--small_blocks", help="number of small convolutional blocks in the network. ~1-4 should work well.", default=4, type=int, choices=[1,2,3,4,5,6,7])
parser.add_argument("-f", "--filters", help="number of filters in each convolution", default=128, type=int)
parser.add_argument("-ti", "--train_integrator", help="train the network", action='store_true')

parser.add_argument("-seq", "--sequence_length", help= "sequence length at inference time. How long a sequence should the networks generate ?", default=30, type=int)
parser.add_argument('-sg', '--graph_saving_freq', help= "save meta graph every n frame. no saves when set to zero", default=5000, type=int)
parser.add_argument('-tb', '--tensorboard_saving_freq', help= "save tensorboard plot every n frame. no saves when set to zero", default=5, type=int)
parser.add_argument('-pd', '--prediction_length', help = "Number of frames to predict", default=30, type=int)
parser.add_argument('-dp', '--deploy_path', default='', help="Alternative dir for inference data")
parser.add_argument('-lr_min', '--min_learn_rate', default=0.0000025, help="Minimum learning rate attained during cosine annealing")
parser.add_argument('-lr_max', '--max_learn_rate', default=0.00005, help="Maximum learning rate attained during cosine annealing")
parser.add_argument('-ep', '--period', default=2500, help="period of cosine annealing")
parser.add_argument('-tri', '--trilinear', action='store_true', help="use tri-linear interpolation for resampling and not nearest neighbour")
parser.add_argument('-mlp', '--encoder_mlp_layers', default = 1, type = int,  help="MLP layers to use on each side of the latent state projection")
parser.add_argument('-sdf', '--sdf_state_size', default = 2, type = int,  help="size of the boundary conditions encoding")
parser.add_argument('-gif', '--gif_saver_f', default = 5000, type = int,  help="Frequency for saving gifs")
parser.add_argument('-b', '--batch_size', default = 1, type=int, help='Batch size for training')
parser.add_argument('-fem', '--use_geo_kernels', action='store_true', help="use tri-linear interpolation for resampling and not nearest neighbour")

parser.add_argument('-g', '--grid_path', help='Path to grid dictionary')
args = parser.parse_args()

config.data_path = args.input_dir
config.resample = args.trilinear
config.param_state_size = args.latent_state_size
config.n_filters = args.filters
config.output_dir = args.output_dir
config.save_freq = args.graph_saving_freq
config.f_tensorboard = args.tensorboard_saving_freq
config.sb_blocks = args.small_blocks
config.batch_size = args.batch_size
config.sequence_length = args.sequence_length
config.alt_dir = args.deploy_path
config.lr_max = args.max_learn_rate
config.lr_min = args.min_learn_rate
config.period = args.period
config.encoder_mlp_layers = args.encoder_mlp_layers
config.sdf_state = args.sdf_state_size
config.save_gif = args.gif_saver_f
config.grid_dir = args.grid_path
config.use_fem = args.use_geo_kernels


if not os.path.isdir(config.output_dir):
    print('WARNING - output dir is not valid. Meta graphs are not saved')
else:

    meta_graphs= os.path.join(config.output_dir, 'saved_graphs')
    tensor_board= os.path.join(config.output_dir, 'saved_tensorboard')

    path_e = os.path.join(config.output_dir, 'saved_graphs/encoder/')
    path_i = os.path.join(config.output_dir, 'saved_graphs/integrator/')

    if not os.path.isdir(path_e):
        os.mkdir(path_e)

    if not os.path.isdir(path_i):
        os.mkdir(path_i)

    if not os.path.isdir(tensor_board):
        os.mkdir(tensor_board)

    config.tensor_board=tensor_board
    config.meta_graphs= meta_graphs
    config.path_e = path_e
    config.path_i = path_i


if not os.path.isdir(config.data_path):
    print('Input dir is not valid')

elif args.train:
    train.train_network()

elif args.train_integrator:
    train.train_integrator()

