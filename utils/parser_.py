import argparse, datetime
from utils.utils_ import *

# parser：
parser = argparse.ArgumentParser(description='GP')

# str
# dataset params:
parser.add_argument('--model_dir', default='Models/', help='folder to output model checkpoints')
# col_labels = ['Element', 'Highest Ratio over Control',
#               'Average Ratio over Control', 'Concentration']
parser.add_argument('--id', default='XXXX', help='name')
parser.add_argument('--col_labels', default=None, help="if excel has col name then use None, else: "
                                                       "['material', 'Elemental proportions', \
                                                        'Mass activity at 1.53V', 'slope relative to Ru']")
# parser.add_argument('--col_labels', default= "['Element', 'Highest Ratio over Control', \
#                                              'Average Ratio over Control', 'Concentration']", help='   ')
parser.add_argument('--data_path', default='data/OER-Summary-LZ.xlsx', help='  ')
parser.add_argument('--model', '--y_col_name', default='Mass-activity-at-1.53V+slope-relative-to-Ru',
                    help='OPT1 for PCE: Highest-Ratio-over-Control  OR  Average-Ratio-over-Control /n \
                          OPT2.1 for OER: {Mass-activity-at-1.53V} /n \
                          OPT2.2 for OER: {slope-relative-to-Ru} /n \
                          OPT2.3 for OER: {Mass-activity-at-1.53V+slope-relative-to-Ru}  /n')
## preprocess params:
parser.add_argument('--PCA_dim_select_method', default='auto', help='Other options: assigned')
## MOBO params:
parser.add_argument('--ref_point', default=None, help='   ')
parser.add_argument('--data_search_space', default='data/SearchSpace_3elems_Ru0.8.pkl', help='  ')

# int
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
## SOBO GP params:
parser.add_argument('--num_restarts', type=int, default=5, help='number of epochs to train')
parser.add_argument('--cycle_num', type=float, default=0, help='cycle_num')
## preprocess params:
parser.add_argument('--Kfold', type=int, default=5, help='number of hard neg in loss')
parser.add_argument('--PCA_dim', type=float, default=0.98, help='input: float or int. n compoents of PCA when input')
## MOBO params:
parser.add_argument('--bs', type=int, default=2048, metavar='BS', help='input batch size for training')
parser.add_argument('--q_num', type=int, default=5, help='output number for each query')
parser.add_argument('--mc_samples_num', type=int, default=2048, help='input num of num_mc_samples for opt')

# float
## SOBO GP params:
parser.add_argument('--ker_lengthscale_upper', type=float, default=25, help='ker.lengthscale upper limit')
parser.add_argument('--ker_var_upper', type=float, default=100, help='ker.variance upper limit')
## crossvalidation params:
parser.add_argument('--split_ratio', type=float, default=0, help='split_ratio=test_size/(test_size+train_size')
# parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
# parser.add_argument('--lr', type=float, default=0.05, help='learning rate')

# bool
parser.add_argument('--is_MOBO', default=False, action='store_true',
                    help='settings of MOBO')
## preprocess params:
parser.add_argument('--use_concentration', default=False, action='store_true',
                    help='turns off flip and 90deg rotation augmentation')
parser.add_argument('--only_use_elem2', default=False, action='store_true',
                    help='turns off flip and 90deg rotation augmentation')
parser.add_argument('--use_MI_filter', default=False, action='store_true',
                    help='turns off flip and 90deg rotation augmentation')
parser.add_argument('--use_y_norm', default=False, action='store_true',
                    help='turns off flip and 90deg rotation augmentation')
parser.add_argument('--use_Xnorm_afterPCA', default=False, action='store_true',
                    help='turns off flip and 90deg rotation augmentation')


def clean_args(args) -> list:
    # for args.model:
    args.model = args.model.replace("-", " ")
    args.model = args.model.split("+")

    return args


def get_args(ipynb=False):
    if ipynb:  # for jupyter so that default args are passed
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
    txt += ['bs:' + str(args.bs)]
    txt += ['seed:' + str(args.seed)]
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

    ## In windows recommend to replace ':' with '='
    model_name = model_name.replace(':', '=')
    args.save_name = model_name

    if model_name in [getbase(c) for c in glob(pjoin(args.model_dir, '*'))]:
        printc.red('WARNING: MODEL', model_name, '\nALREADY EXISTS')
    args = clean_args(args)

    return args

