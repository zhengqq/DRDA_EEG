import logging
import os.path
import time
from collections import OrderedDict
import sys

import numpy as np
import torch.nn.functional as F
from torch import optim

from braindecode.models.deep4 import Deep4Net
from braindecode.datasets.bcic_iv_2b import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import (
    LossMonitor,
    MisclassMonitor,
    RuntimeMonitor,
)
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.signalproc import (
    bandpass_cnt,
    exponential_running_standardize,
)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne

log = logging.getLogger(__name__)


def run_exp(data_folder, subject_id, low_cut_hz, model, cuda):
    ival = [0, 4000] # raw[-500, 4000]

    max_epochs = 1600
    max_increase_epochs = 160
    batch_size = 60
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.2



    marker_def = OrderedDict(
        [
            ("Left Hand", [1]),
            ("Right Hand", [2]),
            # ("Foot", [3]),
            # ("Tongue", [4]),
        ]
    )

    trnData, trnLabel = [], []
    for sess_id_trn in range(1, 4):
        train_filename = "B{:02d}{:02d}T.gdf".format(subject_id, sess_id_trn)
        train_filepath = os.path.join(data_folder, train_filename)
        train_label_filepath = train_filepath.replace(".gdf", ".mat")
        train_loader = BCICompetition4Set2A(
            train_filepath, labels_filename=train_label_filepath
        )
        train_cnt = train_loader.load()

        # Preprocessing

        train_cnt = train_cnt.drop_channels(
            ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
        )
        assert len(train_cnt.ch_names) == 3
        # lets convert to millvolt for numerical stability of next operations
        train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
        train_cnt = mne_apply(
            lambda a: bandpass_cnt(
                a,
                low_cut_hz,
                high_cut_hz,
                train_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            train_cnt,
        )
        train_cnt = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            train_cnt,
        )
        train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
        print(train_set.X.shape)
        trnData.append(train_set.X)
        trnLabel.append(train_set.y)

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.concatenate(trnLabel, axis=0)


    tstData, tstLabel = [], []
    for sess_id_tst in range(4,6):

        test_filename = "B{:02d}{:02d}E.gdf".format(subject_id, sess_id_tst)
        test_filepath = os.path.join(data_folder, test_filename)
        test_label_filepath = test_filepath.replace(".gdf", ".mat")
        test_loader = BCICompetition4Set2A(
            test_filepath, labels_filename=test_label_filepath
        )
        test_cnt = test_loader.load()

        test_cnt = test_cnt.drop_channels( ["EOG:ch01", "EOG:ch02", "EOG:ch03"])
        assert len(test_cnt.ch_names) == 3
        test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
        test_cnt = mne_apply(
            lambda a: bandpass_cnt(
                a,
                low_cut_hz,
                high_cut_hz,
                test_cnt.info["sfreq"],
                filt_order=3,
                axis=1,
            ),
            test_cnt,
        )
        test_cnt = mne_apply(
            lambda a: exponential_running_standardize(
                a.T,
                factor_new=factor_new,
                init_block_size=init_block_size,
                eps=1e-4,
            ).T,
            test_cnt,
        )

        test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)
        tstData.append(test_set.X)
        tstLabel.append(test_set.y)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.concatenate(tstLabel, axis=0)




    saveFlag = True
    if saveFlag:
        import scipy.io as sio
        sio.savemat(str(subject_id) +'.mat', {'trnData': trnData, 'trnLabel':trnLabel, 'tstData': tstData, 'tstLabel': tstLabel})


        print('------------------------------------------------------------------------')
        print(subject_id)
        print(trnData.shape)
        print(tstData.shape)
    else:


        train_set, valid_set = split_into_two_sets(
            train_set, first_set_fraction=1 - valid_set_fraction
        )

        set_random_seeds(seed=20190706, cuda=cuda)

        n_classes = 4
        n_chans = int(train_set.X.shape[1])
        input_time_length = train_set.X.shape[2]
        if model == "shallow":
            model = ShallowFBCSPNet(
                n_chans,
                n_classes,
                input_time_length=input_time_length,
                final_conv_length="auto",
            ).create_network()
        elif model == "deep":
            model = Deep4Net(
                n_chans,
                n_classes,
                input_time_length=input_time_length,
                final_conv_length="auto",
            ).create_network()
        if cuda:
            model.cuda()
        log.info("Model: \n{:s}".format(str(model)))

        optimizer = optim.Adam(model.parameters())

        iterator = BalancedBatchSizeIterator(batch_size=batch_size)

        stop_criterion = Or(
            [
                MaxEpochs(max_epochs),
                NoDecrease("valid_misclass", max_increase_epochs),
            ]
        )

        monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]

        model_constraint = MaxNormDefaultConstraint()

        exp = Experiment(
            model,
            train_set,
            valid_set,
            test_set,
            iterator=iterator,
            loss_function=F.nll_loss,
            optimizer=optimizer,
            model_constraint=model_constraint,
            monitors=monitors,
            stop_criterion=stop_criterion,
            remember_best_column="valid_misclass",
            run_after_early_stop=True,
            cuda=cuda,
        )
        exp.run()
        return exp


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    # Should contain both .gdf files and .mat-labelfiles from competition
    data_folder = "/home/keyzhao/Documents/Data/BCI_IV_2b_gdf/"
    for subject_id in range(1,10):

    # subject_id = 1  # 1-9
        low_cut_hz = 4  # 0 or 4
        model = "shallow"  #'shallow' or 'deep'
        cuda = False
        exp = run_exp(data_folder, subject_id, low_cut_hz, model, cuda)
        # log.info("Last 10 epochs")
        # log.info("\n" + str(exp.epochs_df.iloc[-10:]))
