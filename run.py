import argparse
import config
import train
import deploy
import os

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to training fields / See ... for Houdini based data generator")
parser.add_argument("--output_dir", default='', help="path to training fields / See ... for Houdini based data generator")

parser.add_argument("-mg", "--meta graph_dir", default='', help="graph saving directory")
parser.add_argument("-t", "--train", help="train the network", default=True, type=bool)
parser.add_argument("-lss", "--latent_state_size", type=int, help= "size of the latent state space. Should be 2^n", default=16, choices=[8, 16, 32, 64, 128])
parser.add_argument("-sb", "--small_blocks", help="number of small convolutional blocks in the network. ~1-4 should work well.", default=4, type=int, choices=[1,2,3,4,5,6,7])
parser.add_argument("-f", "--filters", help="number of filters in each convolution", default=128, type=int)
parser.add_argument("-ti", "--train_integrator", help="number of training runs between each integrator training run.", type=int, default=100)
parser.add_argument("-seq", "--sequence_length", help= "sequence length at inference time. How long a sequence should the networks generate ?", default=0, type=int)
parser.add_argument('-sg', '--graph_saving_freq', help= "save meta graph every n frame. no saves when set to zero", default=5000, type=int)
parser.add_argument('-tb', '--tensorboard_saving_freq', help= "save tensorboard plot every n frame. no saves when set to zero", default=0, type=int)
args = parser.parse_args()

config.data_path = args.input_dir
config.param_state_size = args.latent_state_size
config.n_filters = args.filters
config.output_dir = args.output_dir
config.save_freq = args.graph_saving_freq
config.f_tensorboard = args.tensorboard_saving_freq
config.sb_blocks = args.small_blocks

if not os.path.isdir(config.output_dir):
    print('WARNING - output dir is not valid. Meta graphs are not saved')
else:
    meta_graphs= os.path.join(config.output_dir, 'saved_graphs')
    tensor_board= os.path.join(config.output_dir, 'saved_tensorboard')

    if not os.path.isdir(meta_graphs):
        os.mkdir(meta_graphs)
    if not os.path.isdir(tensor_board):
        os.mkdir(tensor_board)

    config.tensor_board=tensor_board
    config.meta_graphs=meta_graphs

if not os.path.isdir(config.data_path):
    print('Input dir is not valid')
elif args.train:
    train.train_network()
