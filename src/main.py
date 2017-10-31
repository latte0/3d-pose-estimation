import argparse
import os
import scipy.misc
import numpy as np

from config.parameter import parameter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')

parser.add_argument('--demo',dest='demo', '', 'demo path')
parser.add_argument('--disable3DHP',dest='disable3DHP', type=bool, default=false, 'help=not validate on h36m')
parser.add_argument('--validH36M', dest='validH36M'', type=bool, default=true, help='not validate on h36m')
parser.add_argument('--valid3DHP',dest='valid3DHP', type=bool, default=true,help= 'validate on mpi-inf-3dhp')
parser.add_argument('--DEBUG', dest='DEBUG', type=int,default=0, help='debug')
parser.add_argument('-display', dest='display' , type=int,default=10, help='Display Loss')
parser.add_argument('-Ratio3D',dest='Ratio3D', type=int,default=5, help='Ratio of 3D data')
parser.add_argument('-expID', dest='expID', default='default', help='Experiment ID')
parser.add_argument('-dataset', dest='dataset', default= 'fusion', help='Dataset choice: mpii | fusion | h36m')
parser.add_argument('-h36mFullTest',dest='h36mFullTest', default=false, help='full test')
    parser.add_argument('-dataDir',dest='dataDir', default= projectDir .. '/data',help= 'Data directory')
parser.add_argument('-mpiiImgDir',dest='mpiiImgDir',  default=paths.concat(os.getenv('HOME'),help='Datasets/mpii/images'))
parser.add_argument('-h36mImgDir', dest='h36mImgDir', default=paths.concat(os.getenv('HOME'),'help=Datasets/Human3.6M/images'))
parser.add_argument('-mpi_inf_3dhpImgDir', dest='mpi_inf_3dhpImgDir',  default=paths.concat(os.getenv('HOME'),help='Datasets/MPI-INF-3DHP/images'))
parser.add_argument('-expDir',  dest='expDir',  projectDir .. '/exp',  help='Experiments directory')
parser.add_argument('-manualSeed', dest='manualSeed', default= -1, help='Manually set RNG seed')
parser.add_argument('-GPU', dest='GPU', default=1, help='Default preferred GPU, if set to -1: no GPU')
parser.add_argument('-finalPredictions',dest='finalPredictions', false, 'help=Generate a final set of predictions at the end of training (default no)')
parser.add_argument('-nThreads',dest='nThreads', type=int,4, help='Number of data loading threads')
parser.add_argument('-gt2D', dest='gt2D', type=bool, default=false, help='gt 2d for constraint')
parser.add_argument('-netType', dest='netType', default='hgreg-3d', help='Options: hg-reg-3d')
parser.add_argument('-loadModel',dest='loadModel', default='none', help='Provide full path to a previously trained model')
parser.add_argument('-continue', dest='continue', type=bool, default=false, help='Pick up where an experiment left off')
parser.add_argument('-branch',dest='branch',  default='none', help='Provide a parent expID to branch off')
parser.add_argument('-task',dest='task',  default='pose-hgreg-3d', help='Network task: pose-3d-reg')
parser.add_argument('-nFeats', dest=''nFeats,  type=int, default=256, help='Number of features in the hourglass')
parser.add_argument('-nStack', dest='nStack',  type=int, default=2, help='Number of hourglasses to stack')
parser.add_argument('-nModules', dest='nModules', type=int, default=2, help='Number of residual modules at each location in the hourglass')
parser.add_argument('-nRegModules',  dest='nRegModules', type=int, default=2, help='Number of residual modules at each location after the hourglass')
parser.add_argument('-validEpoch',  dest='validEpoch', type=int, default=5, help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('-snapshot', dest='snapshot', type=int, default=5, help='How often to take a snapshot of the model (0 = never)')
parser.add_argument('-saveInput', dest='saveInput',  type=bool, default=false, help='Save input to the network (useful for debugging)')
parser.add_argument('-saveHeatmaps', dest='saveHeatmaps',type=bool, default=false, help='Save output heatmaps')
parser.add_argument('-varWeight',  dest='varWeight',  type=float, 0.01, help='Weakly supervised Weight')
parser.add_argument('-regWeight', dest='regWeight',  type=float, 0.1, help='Regression Weight')
parser.add_argument('-PCK_Threshold', dest='PCK_Threshold', type=int, default=150,help= 'PCK_Threshold')
parser.add_argument('-dropLR', dest='dropLR', type=int, default=10000, help='Drop Learning rate')
parser.add_argument('-LR', dest='LR', type=float, type=float, default=2.5e-4, 'Learning rate')
parser.add_argument('-LRdecay', dest='LRdecay',type=float, default=0.0, help='Learning rate decay')
parser.add_argument('-momentum',dest='momentum', type=float,default=0.0, help='Momentum')
parser.add_argument('-weightDecay', dest='weightDecay', type=float, default=0.0, help='Weight decay')
parser.add_argument('-alpha', dest='alpha', type=float, default=0.99, help='Alpha')
parser.add_argument('-epsilon',dest='epsilon', type=float,default=1e-8, help='Epsilon')
parser.add_argument('-crit', dest='crit', default='MSE', help='Criterion type')
parser.add_argument('-optMethod', dest='optMethod', default='rmsprop', help='Optimization method: rmsprop | sgd | nag | adadelta')
parser.add_argument('-threshold', dest='threshold', type=float, default=.001, help='Threshold (on validation accuracy growth) to cut off training early')
parser.add_argument('-nEpochs',dest='nEpochs',type=int, default=60, help='Total number of epochs to run')
parser.add_argument('-trainIters', dest='trainIters', type=int, default=4000, help='Number of train iterations per epoch')
parser.add_argument('-trainBatch', dest='trainBatch', type=int,default=6, help=help='Mini-batch size')
parser.add_argument('-validIters',dest='validIters', type=int,default=2958, help='Number of validation iterations per epoch')
parser.add_argument('-validBatch', dest='validBatch', type=int, default=1, help=help='Mini-batch size for validation')
parser.add_argument('-nValidImgs',dest='nValidImgs', type=int,default=2958,help= 'Number of images to use for validation. Only relevant if randomValid is set to true')
parser.add_argument('-randomValid', dest='randomValid', type=bool,  default=false,help=help= 'Whether or not to use a fixed validation set of 2958 images (same as Tompson et al. 2015)')
parser.add_argument('-inputRes', dest='inputRes', type=int, default=256, help='Input image resolution')
parser.add_argument('-outputRes', dest='outputRes', type=int, default=64, help='Output heatmap resolution')
parser.add_argument('-scale', dest='scale', type=float, default=.25, help='Degree of scale augmentation')
parser.add_argument('-rotate',dest='rotate', type=int,  default=30, help=help='Degree of rotation augmentation')
parser.add_argument('-hmGauss',  dest='hmGauss',  type=int, default=1, help='Heatmap gaussian size')

opts = parser.parse_args()

def main(_):

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)


    #params = parameter(args.lr, args.beta1, args.L1_scale)
