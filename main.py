from easydict import EasyDict
from tensorflow.python.platform import flags
from trainer import main_single

FLAGS = flags.FLAGS

# Distributed training hyperparameters
flags.DEFINE_integer('nodes', 1,
    'number of nodes for training')
flags.DEFINE_integer('gpus', 1,
    'number of gpus per nodes')
flags.DEFINE_integer('node_rank', 0,
    'rank of node')

# Configurations for distributed training
flags.DEFINE_string('master_addr', '8.8.8.8',
    'address of communicating server')
flags.DEFINE_string('port', '10002',
    'port of training')
flags.DEFINE_bool('slurm', False,
    'whether we are on slurm')
flags.DEFINE_bool('repel_im', True,
    'maximize entropy by repeling images from each other')
flags.DEFINE_bool('hmc', False,
    'use the hamiltonian monte carlo sampler')
flags.DEFINE_bool('square_energy', False,
    'make the energy square')
flags.DEFINE_bool('alias', False,
    'make the energy square')

flags.DEFINE_string('sampler','cycsgld',
    'cycsgld, sgld, psgld, asgld, resgld or csgld')
flags.DEFINE_string('dataset','cifar10',
    'cifar10 or celeba')
flags.DEFINE_integer('batch_size', 128, 'batch size during training')
flags.DEFINE_bool('multiscale', False, 'A multiscale EBM')
flags.DEFINE_bool('self_attn', True, 'Use self attention in models')
flags.DEFINE_bool('sigmoid', False, 'Apply sigmoid on energy (can improve the stability)')
flags.DEFINE_bool('anneal', False, 'Decrease noise over Langevin steps')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')
flags.DEFINE_integer('buffer_size', 10000, 'Size of inputs')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_string('entropy', 'kl', 'entropy regularization')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', 0, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_bool('transform', True, 'apply data augmentation when sampling from the replay buffer')
flags.DEFINE_bool('kl', True, 'apply a KL term to loss')
flags.DEFINE_bool('cuda', True, 'move device on cuda')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_integer('ensembles', 1, 'Number of ensembles to train models with')
flags.DEFINE_float('lr', 2e-4, 'Learning for training')
flags.DEFINE_float('kl_coeff', 1.0, 'coefficient for kl')

# EBM Specific Experiments Settings
flags.DEFINE_string('objective', 'cd', 'use the cd objective')

# Setting for MCMC sampling
flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 100.0, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_bool('reservoir', True, 'Use a reservoir of past entires')
flags.DEFINE_float('noise_scale', .23,'Relative amount of noise for MCMC')
flags.DEFINE_float('init_noise', .1,'Relative amount of noise for MCMC')
flags.DEFINE_float('momentum', 0.9,'Relative amount of noise for MCMC')
flags.DEFINE_float('eps', 1e-6,'Relative amount of noise for MCMC')
flags.DEFINE_float('step_size', 10,'Relative amount of noise for MCMC')

# Architecture Settings
flags.DEFINE_integer('filter_dim', 64, 'number of filters for conv nets')
flags.DEFINE_integer('im_size', 32, 'size of images')
flags.DEFINE_bool('spec_norm', False, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('norm', True, 'Use group norm in models norm in models')

# Conditional settings
flags.DEFINE_bool('cond', False, 'conditional generation with the model')
flags.DEFINE_bool('all_step', False, 'backprop through all langevin steps')
flags.DEFINE_bool('log_grad', False, 'log the gradient norm of the kl term')
flags.DEFINE_integer('cond_idx', 0, 'conditioned index')

def main():
    flags_dict = EasyDict()

    for key in dir(FLAGS):
        flags_dict[key] = getattr(FLAGS, key)

    main_single(flags_dict)

if __name__ == "__main__":
    main()