import argparse


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--data', 
        default='~/Afstudeerproject/data/hmdb_videos/jpg',
        help='path to dataset', 
        type=str)
    parser.add_argument(
        '-l', 
        '--list', 
        default='~/Afstudeerproject/data/hmdb_1.txt',
        help='path to video list', 
        type=str)
    parser.add_argument(
        '-j', 
        '--workers', 
        default=12, 
        type=int, 
        metavar='N',
        help='number of data loading workers (default: 4)')
    # Optimization options
    parser.add_argument(
        '--epochs', 
        default=30, 
        type=int, 
        metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--u_lr', 
        '--unsupervised_learning_rate', 
        default=2e-4, 
        type=float,
        metavar='LR', 
        help='initial learning rate')
    parser.add_argument(
        '--u_momentum', 
        default=0.5, 
        type=float, 
        metavar='M',
        help='momentum')
    parser.add_argument(
        '--u_wd',
        '--unsupervised_weight_decay',
        default=0.0, 
        type=float,
        metavar='W', 
        help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument(
        '--pc', 
        '--path_checkpoint', 
        default='~/Afstudeerproject/data/results_timecycle',
        type=str, 
        metavar='PATH',
        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument(
        '--resume', 
        default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    # Miscs
    parser.add_argument(
        '--manualSeed', type=int, help='manual seed')
    parser.add_argument(
        '-e', 
        '--evaluate', 
        dest='evaluate', 
        action='store_true',
        help='evaluate model on validation set')
    parser.add_argument(
        '--pretrained', 
        default='', 
        type=str, 
        metavar='PATH',
        help='use pre-trained model')
    #Device options
    parser.add_argument(
        '--gpu-id', 
        default='0,1,2,3', 
        type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument(
        '--predDistance', 
        default=4, type=int,
        help='predict how many frames away')
    parser.add_argument(
        '--seperate2d', 
        type=int, 
        default=0, 
        help='manual seed')
    parser.add_argument(
        '--batchSize', 
        default=36, 
        type=int,
        help='batchSize')
    parser.add_argument(
        '--T', 
        default=512**-.5, 
        type=float,
        help='temperature')
    parser.add_argument(
        '--gridSize', 
        default=9, 
        type=int,
        help='temperature')
    parser.add_argument(
        '--classNum', 
        default=51, 
        type=int,
        help='temperature')
    parser.add_argument(
        '--lamda', 
        default=0.1, 
        type=float,
        help='temperature')
    parser.add_argument(
        '--pretrained_imagenet', 
        type=str_to_bool, 
        nargs='?', 
        const=True, 
        default=False,
        help='pretrained_imagenet')

    parser.add_argument(
        '--videoLen', 
        default=4, 
        type=int,
        help='')
    parser.add_argument(
        '--frame_gap', 
        default=2, 
        type=int,
        help='')
    parser.add_argument(
        '--hist', 
        default=1, 
        type=int,
        help='')
    parser.add_argument(
        '--optim', 
        default='adam', 
        type=str,
        help='')


    args = parser.parse_args()

    return args
