import argparse, datetime
from utils.utils_ import *

# parser：
parser = argparse.ArgumentParser(description='GP')
# str
parser.add_argument('--model_dir', default='Models/', help='folder to output model checkpoints')
# col_labels = ['Element', 'Highest Ratio over Control',
#               'Average Ratio over Control', 'Concentration']
parser.add_argument('--id', default='XXXX', help='inbatch, epoch')
parser.add_argument('--col_labels', default= "['Element', 'Highest Ratio over Control', \
                                             'Average Ratio over Control', 'Concentration']", help='   ')       #下午可以改为非++的情况看看loss的计算过程
parser.add_argument('--data_path', default='OER-Summary-LZ.xlsx', help='  ')
parser.add_argument('--model', '--y_col_name', default='Average Ratio over Control', help='Highest-Ratio-over-Control  OR  Average-Ratio-over-Control')
parser.add_argument('--PCA_dim_select_method', default='auto', help='Other options: assigned')
# parser.add_argument('--masks_dir', '--masks', default=None , help='')       #'Datasets/AMOS-views/AMOS-masks'
# parser.add_argument('--weight_function', '--wf', default='Hessian',
#                     help='Keypoints are generated with probability ~ weight function. Variants: uniform, Hessian, HessianSqrt, HessianSqrt4')
# int
parser.add_argument('--num_restarts', type=int, default=5, help='number of epochs to train')
parser.add_argument('--Kfold', type=int, default=5, help='number of hard neg in loss')
parser.add_argument('--PCA_dim', type=float, default=0, help='input: float or int. n compoents of PCA when input')
parser.add_argument('--cycle_num', type=float, default=0, help='cycle_num')
# parser.add_argument('--patch_sets', '--psets', '--tracks', type=int, default=30000, help='How many patch sets to generate. Works approximately.')
# parser.add_argument('--bsNum', type=int, default=1400, help='how many batch will ues(only work in FineTune)')
parser.add_argument('--batch_size', '--bs', type=int, default=0, metavar='BS', help='input batch size for training')
# parser.add_argument('--test_batch_size', type=int, default=2048, metavar='BST', help='input batch size for testing (default: 1024)')
# parser.add_argument('--cams_in_batch', '--camsb', type=int, default=5, help='how many cams are source ones for a batch in AMOS')
# parser.add_argument('--min_sets_per_img', '--minsets', type=int, default=-1, help='')
# parser.add_argument('--max_sets_per_img', '--maxsets', type=int, default=1501, help='used in new patch_gen')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
# parser.add_argument('--shear', type=int, default=25, help='augmentation: shear')
# parser.add_argument('--degrees', type=int, default=25, help='augmentation: degrees')
# parser.add_argument('--Npos', type=int, default=2, help='')     ##positive
# float
parser.add_argument('--ker_lengthscale_upper', type=float, default=25, help='ker.lengthscale upper limit') 
parser.add_argument('--ker_var_upper', type=float, default=100, help='ker.variance upper limit') 
# parser.add_argument('--lr', type=float, default=0.05, help='learning rate') 
# parser.add_argument('--lr', type=float, default=0.05, help='learning rate') 
# bool
parser.add_argument('--use_concentration', default=False, action='store_true', help='turns off flip and 90deg rotation augmentation')
# parser.add_argument('--addAP', default=False, action='store_true', help='add AP lsos to standard loss')
# parser.add_argument('--AP_loss', default=False, action='store_true')

def get_args(ipynb=False):
    if ipynb: # for jupyter so that default args are passed
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    printc.yellow('Parsed options:\n{}\n'.format(vars(args)))
    
    # show in txt file(data neme):
    txt = []
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%m%d-%H_%M")
    txt += ['id:' + str(args.id)]
    txt += ['T:' + current_time]
    txt += ['num_ep:' + str(args.num_restarts)]
    if args.cycle_num: txt += ['cycle_num:' + str(args.cycle_num)]
    txt += ['bs:' + str(args.batch_size)]
    txt += ['seed:' + str(args.seed)]
    args.model = args.model.replace("-", " ")
    txt += ['model:' + str(args.model)]
    txt += ['PCA_dim:' + str(args.PCA_dim)]
    txt += ['PCA_dim_select_method:' + args.PCA_dim_select_method]
    txt += ['seed:' + str(args.seed)]
    # txt += ['Data_path:' + args.data_path]
    model_name = '_'.join([str(c) for c in txt])
    # if args.Kfold: txt += ['Kfold:' + str(args.Kfold)]
    # if args.ker_lengthscale_upper: txt += ['ker_lengthscale_upper:' + str(args.ker_lengthscale_upper)]
    # if args.ker_var_upper: txt += ['ker_var_upper:' + str(args.ker_var_upper)]
    # if args.use_concentration: txt += ['UseConcentrationCol ']
    args.save_name = model_name

    if model_name in [getbase(c) for c in glob(pjoin(args.model_dir, '*'))]:
        printc.red('WARNING: MODEL',model_name,'\nALREADY EXISTS')
    
    return args

