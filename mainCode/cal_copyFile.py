import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_name', default=None)
parser.add_argument('--log_path', default='')
parser.add_argument('--fold', default=5,type=int)
args = parser.parse_args()

log_dir = args.log_path + '/' + args.exp_name
fold = args.fold

avg_acc=0
avg_kappa = 0
for i in range(fold):

    dir_name = log_dir + '/' + str(i+1)
    file_name = dir_name + '/result.txt'
    target_dir = log_dir + '/'
    log_name = dir_name + '/' + 'training_log.txt'

    l = open(file_name).readlines()

    with open(target_dir + '/result.txt', 'a+') as f:
        f.write('Round %2d: %s\n' % (i + 1, l[-1][:-1]))

    avg_acc += float(l[-1][18:24])
    avg_kappa += float(l[-1][34:40])


avg_acc /= fold
avg_kappa /= fold
with open(log_dir + '/result.txt', 'a+') as f:
    f.write('Average accuracy: %.4f, kappa: %.4f\n' % (avg_acc, avg_kappa))
