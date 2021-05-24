exp='bci_0-4_s1_t1_adv10_c0_1'

mkdir ./Model_and_Result/${exp}

cp $(basename $0) ./Model_and_Result/${exp}/

for runTime in {1..5}
do

for i in {1..9}
do

    python BCI_DA.py --exp_name ${exp}'/'${runTime} --lr 0.0002 --max_epoch 200 --criteria val_loss  --loadN 1 --stop_tolerance 10 --classifier signal_da_fc64 --subject $i --batch_size 64 --data_len data_0-4 --w_adv 10 --w_t 1 --w_s 1 --w_c 0
    
    echo 'Subject ' $i 'done'
    echo ''
    echo ''
    echo ''
    echo '------------------------------------------------------------------------------------------' 

done

echo 'Runtime ' $runTime 'done'
echo '---------------------------------------------------------------------------------------'

done


