def add_training_args(parser):

    parser.add_argument('--rand_seed', '-rs', type=int,  help='random seed')
    parser.add_argument('--job_id', '-jb', type=int, default=0,help='job_id')
    parser.add_argument('--exp_name', '-exp_name', type=str, default='test_experiment', help='name of experiment')
    
    #GPU gestion
    parser.add_argument('--gpu_ids', '-g', type=int, default=[],nargs='+', help='ids of GPUs to use')
    parser.add_argument('--multi_gpu','-multi_gpu',action='store_true',help='use distributed training')
        
    #Dataset
    parser.add_argument('--n_classes', '-nc', type=int, default=10,help='number of classes')
    parser.add_argument('--batch_size', '-bsz', type=int, default=32, help='batch size')
    parser.add_argument('--source', '-sc', type=str, default='mnist10k', help='souce domain for training')
    parser.add_argument('--inter','-inter',action='store_true', help='inter or intra domain generalization for hhar')
    parser.add_argument('--source_localisation','-sloc',default='H', help='location of captor for intra domain generalization for hhar')
    parser.add_argument('--multi_aug', '-ma', action='store_true', help='strong data augmentations')
    
    #Network
    parser.add_argument('--net', '-net', type=str, default='digit',  help='network')
    parser.add_argument('--pretrained', '-pt', action='store_true',  help='use pretrained network')
    
    #Training parameter : Optimizer, Scheduler, Training schedule
    parser.add_argument('--lr', '-lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--nesterov', '-nest', action='store_true', help='use nesterov momentum')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', '-mmt', default=0.9, type=float, help='momentum')
    parser.add_argument('--scheduler', '-sch', type=str, default='', help='type of lr scheduler, StepLR/MultiStepLR/CosLR')
    parser.add_argument('--step_size', '-stp', type=int, default=30, help='fixed step size for StepLR')
    parser.add_argument('--milestones', '-milestones', type=int, nargs='+', help='milestone for MultiStepLR')
    parser.add_argument('--gamma', '-gm', type=float,  default=0.2, help='reduce rate for step scheduler')
    parser.add_argument('--power', '-power', default=0.9, help='power for poly scheduler')
    parser.add_argument('--n_epoch', '-n_epoch', type=int, default=[100,],nargs='+', help='number of trainning epochs (liste for the case of iterative pruning with fine-tuning and learning rate rewinding)')
    parser.add_argument('--n_iter', '-ni', type=int, default=[], nargs='+',help='number of total training iterations (liste for the case of iterative pruning with fine-tuning and learning rate rewinding)')
    parser.add_argument('--val_iter', '-vi', type=int, default=100, help='number of training iterations between two validations')
    
    #Logs
    parser.add_argument('--verbose', '-verbose', action='store_true', help='verbose to print bar')
    parser.add_argument('--save_model', '-sm', action='store_true', help='save model at the end of training')
    parser.add_argument('--metric', '-metric', type=str, default=["accuracy",], nargs='+',help='metric used')
                                          
    

def add_rand_layer_args(parser):
    parser.add_argument('--rand_conv', '-rc', action='store_true', help='use random layers')
    parser.add_argument('--channel_size', '-chs', type=int, default=3,
                        help='Number of output channel size  random layers, '
                        )
    parser.add_argument('--kernel_size', '-ks', type=int, default=[3,], nargs='+',
                        help='kernal size for random layer, could be multiple kernels for multiscale mode')
    parser.add_argument('--rand_bias', '-rb', action='store_true',
                        help='add random bias in convolution layer')
    parser.add_argument('--distribution', '-db', type=str, default='kaiming_normal',
                        help='distribution of random sampling')
    parser.add_argument('--clamp_output', '-clamp', action='store_true',
                        help='clamp value range of randconv outputs to a range (as in original image)'
                        )
    parser.add_argument('--mixing', '-mix', action='store_true',
                        help='mix the output of rand conv layer with the original input')
    parser.add_argument('--identity_prob', '-idp', type=float, default=0.0,
                        help='the probability that the rand conv is a identity map, '
                             'in this case, the output and input must have the same channel number')
    parser.add_argument('--multi_scale', '-ms', type=str, nargs='+',
                        help='multiscale settings, e.g. \'3-3\' means kernel size 3 with output channel size 3')
    parser.add_argument('--rand_freq', '-rf', type=int, default=1,
                        help='frequency of randomize weights of random layers (every n steps)')
    parser.add_argument('--train-all', '-ta', action='store_true',
                        help='train all random layers, use for ablation study when the network is modified')
    parser.add_argument('--consistency_loss', '-cl', action='store_true',
                        help='use invariant loss to enforce similar predictionso on different augmentation of the same input')
    parser.add_argument('--consistency_loss_w', '-clw', type=float, default=1.0,
                        help='weight for invariant loss')
    parser.add_argument('--augmix', '-am', action='store_true',
                        help='aug_mix mode, only the raw data is used to compute classfication loss')
    parser.add_argument('--n_val', '-nv', type=int, default=1,
                        help='repeat validation with different randconv')
    parser.add_argument('--val_with_rand', '-vwr', action='store_true',
                        help='validation with random conv;'
                        )
                        
                        
def add_pruning(parser):

    parser.add_argument('--pruning','-pruning',action='store_true',help='use a pruning algorithm to prune model')
    parser.add_argument('--pruning_algorithm','-p_algs',type=str,default='L1',help='define the pruning algorithm')
    parser.add_argument('--pruning_schedule', '-ps', type=int, default=[0,], nargs='+',help='pruning schedule (proportion of pruned parameters)')
    parser.add_argument('--p_reset','-p_reset',action='store_true',help='Choose if the pruning algorithm can unpruned pruned weights')
    parser.add_argument('--p_layerwise','-p_layerwise',action='store_true',help='Choose if the pruning algorithm compute threshold layerwise')
    parser.add_argument('--prune_iterative','-p_iter',type=int,default=1,help='define iterative in pruning algorithm')
    parser.add_argument('--prune_last','-prune_last',action='store_true',help='Choose if the pruning algorithm prune the last layer')
    parser.add_argument('--snip_mode','-snip_mode',type=str,default='rand_conv',help='define the mode to compute gradients in SNIP algorithm')
    parser.add_argument('--snip_iter','-snip_iter',type=int,default=3,help='define rand_iter in SNIP algorithm')
    parser.add_argument('--snip_batch','-snip_batch',type=int,default=0,help='define mini_batches in SNIP algorithm')
    parser.add_argument('--rewinding','-rewinding',action='store_true',help='Choose if rewinding will be used')
    parser.add_argument('--rewinding_epoch','-re',type=int,default=0,help='Epoch for the rewinding')
    parser.add_argument('--finetuning','--finetuning',action='store_true',help='Choose if learning rate rewinding is not used (finetuning)')
    
                        
def get_exp_name(args):
    exp_name = "".join([
        args.net,'-{}'.format(args.source),'-seed{}'.format(args.rand_seed),
        '-MultiAug' if args.multi_aug else '',
        '-clampOutput' if args.clamp_output and args.rand_conv else '',
        '-randConv' if args.rand_conv else '',
        '-ch{}'.format(args.channel_size) if args.rand_conv else '',
        '-{}'.format(args.distribution) if args.distribution and args.rand_conv else '',
        '-kz{}'.format('_'.join(str(k) for k in args.kernel_size)) if args.kernel_size and args.rand_conv else '',

        '-randbias'.format(args.rand_bias) if args.rand_bias and args.rand_conv else '',
        '-mixing' if args.mixing and args.rand_conv else '',
        '-idprob_{}'.format(args.identity_prob) if args.identity_prob > 0 and args.rand_conv else '',
        '-freq{}'.format(args.rand_freq) if (args.rand_conv and not args.train_all) else '',
        '-{}cons_{}'.format('augmix-' if args.augmix else '', args.consistency_loss_w)
        if args.consistency_loss and args.rand_conv else '',
        '-val_rand{}'.format(args.n_val)
        if args.val_with_rand and args.rand_conv else '',
        '-lr{}'.format(args.lr),
        '-batch{}'.format(args.batch_size),
        '-SGD-{}mom{}-wd{}'.format('nesterov' if args.nesterov else '', args.momentum, args.weight_decay) if args.SGD else '',
        '-{}Schd_step{}_gamma{}'.format(args.scheduler, args.step_size, args.gamma) if args.scheduler == 'StepLR' else '',
        '-{}Schd_{}_gamma{}'.format(args.scheduler, '_'.join([str(i) for i in args.milestones]), args.gamma) if args.scheduler == 'MultiStepLR' else '',
        '-{}Schd'.format(args.scheduler) if args.scheduler == 'CosLR' else '',
        '-{}ep'.format(args.n_epoch) if len(args.n_iter)<1 else '-{}iters'.format(args.n_iter),
        '-pruning' if args.pruning else '',
        '-palgs_{}'.format(args.pruning_algorithm) if args.pruning else '',
        '-pschedule_{}'.format(args.pruning_schedule) if args.pruning else '',
        '-p_reset' if args.p_reset else '',
        '-p_layerwise' if args.p_layerwise else '',
        '-prune_last' if args.prune_last else '',
        '-snip_mode_{}'.format(args.snip_mode) if args.pruning else '',
        '-snip_iter_{}'.format(args.snip_iter) if args.pruning else '',
        '-snip_batch_{}'.format(args.snip_batch) if args.pruning else '',
        '-p_iter_{}'.format(args.prune_iterative),
        '-rewinding' if args.rewinding else '',
        '-r_epochs_{}'.format(args.rewinding_epoch) if args.rewinding else '',
        '-finetuning' if args.finetuning else '',
    ])
    return exp_name