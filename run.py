import argparse
import config
import train
import os
import util
import inference


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
parser.add_argument('-lr_min', '--min_learn_rate', type=float, default=0.0000025, help="Minimum learning rate attained during cosine annealing")
parser.add_argument('-lr_max', '--max_learn_rate', type=float, default=0.00005, help="Maximum learning rate attained during cosine annealing")
parser.add_argument('-ep', '--period', default=2500, help="period of cosine annealing")
parser.add_argument('-tri', '--trilinear', action='store_true', help="use tri-linear interpolation for resampling and not nearest neighbour")
parser.add_argument('-mlp', '--encoder_mlp_layers', default = 1, type = int,  help="MLP layers to use on each side of the latent state projection")
parser.add_argument('-sdf', '--sdf_state_size', default = 8, type = int,  help="size of the boundary conditions encoding")
parser.add_argument('-gif', '--gif_saver_f', default = 5000, type = int,  help="Frequency for saving gifs")
parser.add_argument('-b', '--batch_size', default = 1, type=int, help='Batch size for training')
parser.add_argument('-fem', '--use_differential_kernels', action='store_true', help="use fem layers")
parser.add_argument('-cv', '--convolution', action='store_true', help="use convolutions all the way through the autoencoder")
parser.add_argument('-fem_loss', '--fem_difference', action='store_true', help="use the fem differentials as loss metric")
parser.add_argument('-clear', '--clear', action='store_true', help="clear graphs and test fields in native dirs")

parser.add_argument('-g', '--grid_path', default ='',  help='Path to grid dictionary')
args = parser.parse_args()

config.data_path = args.input_dir


if os.path.isdir(config.data_path):
    util.create_dirs(args.clear)
else:
    print('Input dir is not valid')

if not os.path.isdir(config.output_dir):
    print('WARNING - output dir is not valid. Meta graphs are not saved')
    exit()

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


if os.path.isdir(args.grid_path):
    config.grid_dir = args.grid_path

config.use_fem = args.use_differential_kernels
config.fem_loss = args.fem_difference
config.conv = args.convolution

if args.train:
    train.train_network()

elif args.train_integrator:
    train.train_integrator()
else:
    print('Inference AE with random field')
    inference.restore_ae(config.path_e)



