import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("data_input_dir", help="path to training fields / See ... for Houdini based data generator")
parser.add_argument("-mg", "--meta graph_dir", default='', help="graph saving directory")
parser.add_argument("-t", "--train", help="train the network", default=True, type=bool)
parser.add_argument("-lss", "--latent_state_size", type=int, help= "size of the latent state space. Should be 2^n", default=8, choices=[8, 16, 32, 64, 128])
parser.add_argument("-sb", "--small_blocks", help="number of small convolutional blocks in the network. ~1-4 should work well.", default=4, type=int, choices=[1,2,3,4,5,6,7])
parser.add_argument("-f", "--filters", help="number of filters in each convolution", default=128, type=int)
parser.add_argument("-ti", "--train_integrator", help="number of training runs between each integrator training run.", type=int, default=100)
parser.add_argument("-seq", "--sequence_length", help= "sequence length at inference time. How long a sequence should the networks generate ?", default=0, type=int)
args = parser.parse_args()

config.data_path = args.data_input_dir
config.param_state_size = args.latent_state_size
config.n_filters = args.filters
