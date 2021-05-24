import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_name', default='siamese/1')
parser.add_argument('--log_path', default='')
parser.add_argument('--subnum', default=9, type=int)
args = parser.parse_args()

exp_name = args.exp_name
log_dir = args.log_path + '/' + exp_name

# '/home/keyzhao/pythonScript/supervised-mi-local-sys-for-python_tf/AlgorithmImplement/classifier_params/' + exp_name


if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

avg_acc = 0
avg_kappa = 0

for i in range(args.subnum):
    dir_name = 'Model_and_Result/' + exp_name + '/' + 's' + str(i + 1)
    file_name = dir_name + '/models/checkpoint'
    target_dir = log_dir+'/s' +str(i+1)  + '/'
    log_name = dir_name + '/' + 'training_log.txt'

    os.makedirs(target_dir)
    
    s = open(file_name).read()

    pos = [a for a,i in enumerate(s) if i == '"']
    model = s[pos[0]+1: pos[1]]

    l = open(log_name).readlines()

    with open(log_dir + '/result.txt', 'a+') as f:
        f.write('Subject %2d: %s\n' % (i+1, l[-1][38:83]))

    avg_acc += float(l[-1][62:68])
    avg_kappa += float(l[-1][77:83])

    shutil.copy(dir_name + '/models/' + model + '.index', target_dir)
    shutil.copy(dir_name + '/models/' + model + '.data-00000-of-00001', target_dir)
    shutil.copy(dir_name + '/models/' + model + '.meta', target_dir)
    shutil.copy(dir_name + '/models/' + 'checkpoint', target_dir)
    shutil.copy(dir_name + '/' + 'training_log.txt', target_dir)

avg_acc /= args.subnum
avg_kappa /= args.subnum

#import pdb
#pdb.set_trace()
with open(log_dir + '/result.txt', 'a+') as f:
    f.write('Average accuracy: %.4f, kappa: %.4f\n' % (avg_acc, avg_kappa))
