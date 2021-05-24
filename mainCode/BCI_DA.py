import numpy as np
import scipy.io as sio
import tensorflow as tf
import net
import os
import time
from tensorflow.python.training import training_util
import random
import sklearn
import dataFile
import tensorflow.contrib.slim as slim
from tensorflow import keras
import matplotlib.pyplot as plt
import shutil
import argparse
import ast
import utils
import warnings

np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
## =================================== argparse ========================================
parser = argparse.ArgumentParser()
# parser.add_argument('--model', default='test')
parser.add_argument('--dataset', default='B')
parser.add_argument('--subject', default=5, type=int)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--max_epoch', default=200, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--exp_name', default='test')
parser.add_argument('--aug', default=False,type=ast.literal_eval)
parser.add_argument('--classifier', default='signal_da_fc64')
parser.add_argument('--criteria', default='val_loss')
parser.add_argument('--loadN', default='1')
# parser.add_argument('--dataload', default='dataB')
parser.add_argument('--stop_tolerance', default=100, type=int)
parser.add_argument('--data_len', default='data_0-4')
parser.add_argument('--w_adv', default='0.01',type=float)
parser.add_argument('--w_t', default='1', type=float)
parser.add_argument('--w_s', default='1', type=float)
parser.add_argument('--w_c', default='0.05', type=float)

args = parser.parse_args()

print(args)

## =================================== path set ========================================

load_model = None#'test_init'
save_model = True


# import pdb
# pdb.set_trace()

exp_name = args.exp_name

try:
    model = args.model
except:
    model = 's' + str(args.subject)

result_dir = 'Model_and_Result' + '/' + exp_name + '/' + model
model_directory = result_dir + '/models'

## data path

# mine
dataset = args.dataset
dataPath = args.subject

if tf.gfile.Exists(result_dir):
    tf.gfile.DeleteRecursively(result_dir)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

shutil.copy(__file__, result_dir)
shutil.copy('net.py', result_dir)
shutil.copy('dataFile.py', result_dir)
shutil.copy('loadData.py', result_dir)

# os.system('cp {} {}'.format(__file__, result_dir))
# os.system('cp {} {}'.format('net.py', result_dir))

with open(result_dir + '/training_log.txt', 'w') as f:
    f.close()


## =========================== parameters set =========================

# channel_size = 240
# time_size = 150

lr = args.lr
# beta1 = 0.5
max_epoch = args.max_epoch
batch_size = args.batch_size

stop_criteria = args.criteria

# cls_num = 2


## ============================= model ===============================

if args.classifier == 'signal':
    generator = net.signal_da#spectrogram_net
elif args.classifier == 'signal_more':
    generator = net.signal_more
elif args.classifier == 'signal_da':
    generator = net.signal_siamese_da
elif args.classifier == 'signal_da_fc64':
    generator = net.signal_siamese_da_fc64



discriminator = net.discriminator


# ================================   data   ==========================================================

if dataset == 'A':
    if dataPath < 6:
        path = './data/dataA/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
    else:
        path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
elif dataset == 'B':
    if dataPath < 8:
        path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
    else:
        path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'

# trnData, trnLabel, tstData, tstLabel = dataFile.dataB_valSet(path, loadN=args.loadN, aug=args.aug)

# sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.data_multitask_da(aug=args.aug, dataset=dataset, loadN=args.loadN, subject=args.subject)
sourceData, sourceLabel, targetData, targetLabel, tstData, tstLabel = dataFile.bciiv2a_multitask_da(subject=args.subject, data_len=args.data_len)

# trnData = trnData.astype('float32')
# tstData = tstData.astype('float32')

print('Source set label and proportion:\t', np.unique(sourceLabel, return_counts=True))
print('Target set label and proportion:\t', np.unique(targetLabel, return_counts=True))
print('Val   set label and proportion:\t', np.unique(tstLabel, return_counts=True))

cls_num = len(np.unique(targetLabel))
# cls_num = 10
dataSize = targetData.shape
channel_size = dataSize[1]
time_size = dataSize[2]
try:
    depth_size = dataSize[3]
except:
    depth_size = 1

targetLabel = keras.utils.to_categorical(targetLabel, num_classes=cls_num)
sourceLabel = keras.utils.to_categorical(sourceLabel, num_classes=cls_num)
tstLabel = keras.utils.to_categorical(tstLabel, num_classes=cls_num)



# ===================================== model definition ====================================

tf.reset_default_graph()

input_layer = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32)
label_layer = tf.placeholder(shape=[None, cls_num], dtype=tf.float32)
# is_training = tf.placeholder(shape=[], dtype=tf.bool)
input_layer_s = tf.placeholder(shape=[None, channel_size, time_size, depth_size], dtype=tf.float32)
label_layer_s = tf.placeholder(shape=[None, cls_num], dtype=tf.float32)


predict, prob, feat, net2 = generator(input_layer, channel_size, cls_num)
predict_s, prob_s, feat_s, net2_s = generator(input_layer_s, channel_size, cls_num, reuse=True)

Dx, Dx_logits = discriminator(feat)
Dg, Dg_logits = discriminator(feat_s, reuse=True) #fake
tf.argmax(label_layer, axis=1)

## ============================ loss function and optimizer =======================


with tf.name_scope('d_loss'):
#    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logits, labels=tf.ones_like(Dx)))
#    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logits, labels=tf.zeros_like(Dg)))
#    d_loss = 0.5*d_loss_real + 0.5*d_loss_fake

    d_loss = tf.reduce_mean(tf.square(Dx_logits - 1) + tf.square(Dg_logits)) / 2


with tf.name_scope('g_loss'):

    g_loss_adv = tf.reduce_mean(tf.square(Dg_logits - 1)) / 2
   # g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logits, labels=tf.ones_like(Dg)))
   
    g_loss_ce_t = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_layer)#+ tf.losses.get_regularization_loss()
    g_loss_ce_s = tf.losses.softmax_cross_entropy(logits=predict_s, onehot_labels=label_layer_s)
    
    g_loss_center, centers = utils.get_center_loss(net2, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num, name='centers')
#    g_loss_center_s, centers_s = utils.get_center_loss(feat_s, tf.argmax(label_layer, axis=1), alpha=0.5, num_classes=cls_num, name='centers_s')


    g_loss = args.w_adv * g_loss_adv + args.w_t*g_loss_ce_t + args.w_s*g_loss_ce_s + args.w_c *g_loss_center

    loss_val = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=label_layer)#+ tf.losses.get_regularization_loss()


# split the variable for two differentiable function
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]


# optimizer
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step,
                                           10, 0.99999, staircase=True)

with tf.name_scope('train'):
    # d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*0.4).minimize(d_loss, global_step=global_step, var_list=d_vars)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, global_step=global_step, var_list=d_vars)
#    with tf.control_dependencies([centers_update_op]):
    g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(g_loss, var_list=g_vars)


## ============================ train phase ======================================
# init = tf.global_variables_initializer()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=None)

sess.run(init)


## ============================ save initialization ==============================

if load_model:
    # ckpt = tf.train.get_checkpoint_state('Model_and_Result/' + load_model + '/models')
    # saver.restore(sess, ckpt.model_checkpoint_path)
    ckpt = 'Model_and_Result/' + load_model + '/' + 'models/' + 'model.ckpt-0'
    saver.restore(sess, ckpt)
    print('Saved model loaded')
elif save_model:
    saver.save(sess, model_directory + '/model.ckpt-' + str(0))
    print('Begining model saved')

## ============================ start training =================================

if (len(sourceData) % batch_size) == 0:
    maxIter = int(len(sourceData) / batch_size)
else:
    maxIter = int(len(sourceData) / batch_size + 1)

best_loss = np.inf
stop_step = 0
early_stop_tolerance = args.stop_tolerance
best_acc = 0

stime = time.time()
for epoch in range(max_epoch):

    samList_s = list(range(len(sourceData)))
    samList = list(range(len(targetData)))
    # random.seed(2019)
    random.shuffle(samList_s)


    print('epoch:', epoch+1)
    trainLoss = 0
    valLoss = []
    lossAll, n_exp = 0, 0
    predT = []
    idxEpoch = []
    trainDLoss = 0



    for itr in range(maxIter):
        batch_trn_idx = samList_s[batch_size*itr : batch_size*(itr+1)]
        signalTrain_s = sourceData[batch_trn_idx]

        # random.shuffle(samList)
        batch_idx = np.random.choice(targetData.shape[0], len(batch_trn_idx))
        signalTrain = targetData[batch_idx]

        if len(signalTrain.shape) != 4:
            signalTrain = np.expand_dims(signalTrain, axis=-1)
            signalTrain_s = np.expand_dims(signalTrain_s, axis=-1)

        labelTrain_s = sourceLabel[batch_trn_idx]
        labelTrain = targetLabel[batch_idx]
        # labelTrain = np.expand_dims(labelTrain, axis=1)

        feed_dict = {input_layer: signalTrain, input_layer_s: signalTrain_s, label_layer: labelTrain, label_layer_s: labelTrain_s}

        _, d_loss_value = sess.run([d_optimizer, d_loss], feed_dict=feed_dict)

        # _ = sess.run([g_optimizer], feed_dict=feed_dict)
        _, loss_value, predictV, predictV_s = sess.run([g_optimizer, g_loss, prob, prob_s], feed_dict=feed_dict)

        if int(itr % int(maxIter / 1)) == 10000000:
            print('[Epoch: %2d / %2d] [%4d] loss: %.4f\n'
                % (epoch+1, max_epoch, itr, loss_value))
            with open(result_dir + '/training_log.txt', 'a') as text_file:
                text_file.write(
                    '[Epoch: %2d / %2d] [%4d] loss: %.4f\n'
                    % (epoch+1, max_epoch, itr, loss_value))

        trainLoss = trainLoss + loss_value * len(signalTrain)
        predT.extend(predictV)
        idxEpoch.extend(batch_idx)
        trainDLoss = trainDLoss + d_loss_value * len(signalTrain)


    trainLoss = trainLoss / len(sourceData)
    aa = np.array(predT)
    accT = sklearn.metrics.accuracy_score(np.argmax(targetLabel[idxEpoch], 1), np.argmax(aa, 1))
    trainDLoss = trainDLoss / len(sourceData)


    signalTest = tstData
    if len(signalTest.shape) != 4:
        signalTest = np.expand_dims(signalTest, axis=-1)

    labelTest = tstLabel
    feed_dict = {input_layer: signalTest, label_layer: labelTest}

    val_loss_value, predE = sess.run([loss_val, prob], feed_dict=feed_dict)

    acc = sklearn.metrics.accuracy_score(np.argmax(labelTest,1), np.argmax(predE,1))
    kappa = sklearn.metrics.cohen_kappa_score(np.argmax(labelTest,1), np.argmax(predE,1))

    print('[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'
          % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss, val_loss_value, acc, kappa))
    with open(result_dir + '/training_log.txt', 'a') as text_file:
        text_file.write("[EPOCH: %2d / %2d (global step = %d)] train loss: %.4f, accT: %.4f, Dloss: %.4f; valid loss: %.4f, acc: %.4f, kappa: %.4f'\n"
                        % (epoch + 1, max_epoch, training_util.global_step(sess, global_step), trainLoss, accT, trainDLoss, val_loss_value, acc, kappa))


    # save model
    checkpoint_path = model_directory + '/' + 'model.ckpt'
    saver.save(sess, checkpoint_path, global_step=global_step)


    if stop_criteria == 'val_loss':
        if val_loss_value < best_loss-0.0002:
            best_loss = val_loss_value
            best_acc = acc
            best_kappa = kappa
            stop_step = 0
            best_epoch = epoch
            best_global_step = training_util.global_step(sess, global_step)
        else:
            stop_step += 1
            if stop_step > early_stop_tolerance:
                # print('Early stopping is trigger at epoch: %2d. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)'
                #       %(epoch+1, best_loss, best_acc, best_epoch+1, best_global_step))
                #
                # with open(result_dir + '/training_log.txt', 'a') as text_file:
                #     text_file.write(
                #         'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f at epoch %2d (step = %2d)\n'
                #         % (best_loss, best_acc, best_epoch+1, best_global_step))
                # s = open(model_directory + '/checkpoint').read()
                # s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
                # f = open(model_directory + '/checkpoint', 'w')
                # f.write(s)
                # f.close()
                break
    elif stop_criteria == 'val_acc':
        if (best_acc < acc) or (abs(best_acc - acc) < 0.0001 and val_loss_value < best_loss):
            best_acc = acc
            best_loss = val_loss_value
            best_kappa = kappa
            best_epoch = epoch
            best_global_step = training_util.global_step(sess, global_step)




print('Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)'
      %(best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))
with open(result_dir + '/training_log.txt', 'a') as text_file:
    text_file.write(
        'Early stopping is trigger. ----->>>>> Best loss: %.4f, acc: %.4f, kappa: %.4f at epoch %2d (step = %2d)\n'
        % (best_loss, best_acc, best_kappa, best_epoch+1, best_global_step))
s = open(model_directory + '/checkpoint').read()
s = s.replace('model_checkpoint_path: "model.ckpt-' + str(training_util.global_step(sess, global_step)) + '"', 'model_checkpoint_path: "model.ckpt-' + str(best_global_step) +'"')
f = open(model_directory + '/checkpoint', 'w')
f.write(s)
f.close()



sess.close()
print('finished')
