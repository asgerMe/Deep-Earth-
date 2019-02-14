# Deep-Earth-
Ongoing project for visco-elasitc flow prediction with convolutional autoencoders

Network:
Our network consists of a convolutional deep autoencoder and an integrator network to evolve the latent representation of the physical fields over several time steps. 

Data Generation:
We generate training data with a Houdini based digital asset (.hipnc). The  non-commercial version of Houdini is required (https://www.sidefx.com/download/) to run the SIMs. For now, our network is trained on parameterized fluid simulations to benchmark against existing networks. Additionally, these fluid Sims are easy and fast to run on a regular laptop.    

