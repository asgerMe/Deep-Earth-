import os
import config

if(os.path.isdir(config.data_path)):
    output_path = os.path.join(config.data_path, 'network_output')

    grid_path = os.path.join(config.data_path, 'grid')
    config.grid_dir = grid_path
    graphs_path = os.path.join(output_path, 'graphs')

    integrator_graph = os.path.join(graphs_path, 'integrator')
    autoencoder_graph = os.path.join(graphs_path, '/autoencoder')
    test_fields = os.path.join(output_path, 'fields')
    gifs = os.path.join(output_path, 'gifs')

    tensorboard_path = os.path.join(output_path, 'tensorboard')

    print(output_path)
    print('creating', graphs_path)
    if not os.path.isdir(output_path):
        print('Creating Folders')
        os.mkdir(output_path)
    config.output_dir = output_path

    if not os.path.isdir(grid_path):
        print('creating', grid_path)
        os.mkdir(grid_path)
    config.grid_dir = grid_path

    if not os.path.isdir(graphs_path):
        print('creating', graphs_path)
        os.mkdir(graphs_path)

    if not os.path.isdir(integrator_graph):
        print('creating', integrator_graph)
        os.mkdir(integrator_graph)
    config.path_i = integrator_graph

    if not os.path.isdir(autoencoder_graph):
        print('creating', autoencoder_graph)
        os.mkdir(autoencoder_graph)
    config.path_e = autoencoder_graph

    if not os.path.isdir(test_fields):
        print('creating', test_fields)
        os.mkdir(test_fields)
    config.test_field_path = test_fields

    if not os.path.isdir(gifs):
        print('creating', gifs)
        os.mkdir(gifs)
    config.gif_path = gifs

    if not os.path.isdir(tensorboard_path):
        print('creating', tensorboard_path)
        os.mkdir(tensorboard_path)
    config.tensor_board = tensorboard_path