export CUDA_VISIBLE_DEVICES=0

usePreprocess=softmax

debug=0
epoch=10
itr=3
patience=3








root_path_name=./dataset/Mix/
data_path_name=Mix.csv
model_id_name=Mix
data_name=Mix
enc_in=356
learning_rate=0.0005
lradj=type1
batch_size=16
seq_len=360
H_order=10
e_layers=3
d_model=512
for pred_len in 96 192
do 
for model_name in FITS Linear DLinear iTransformer
do 
for VE_len in 0
do 
for ParamRatio in 1
do 
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --H_order $H_order \
        --e_layers $e_layers \
        --c_out $enc_in \
        --d_model $d_model \
        --d_ff $d_model \
        --VE_len $VE_len \
        --ParamRatio $ParamRatio \
        --usePreprocess $usePreprocess \
        --des 'Exp' \
        --train_epochs $epoch \
        --patience $patience \
        --learning_rate $learning_rate \
        --lradj $lradj \
        --batch_size $batch_size \
        --itr $itr \
        --debug $debug
done
done
done
done








root_path_name=./dataset/Mix/
data_path_name=Mix.csv
model_id_name=Mix
data_name=Mix
enc_in=356
learning_rate=0.0005
lradj=type1
batch_size=16
seq_len=360
H_order=10
e_layers=3
d_model=512
for pred_len in 96 192
do 
for model_name in FITS Linear DLinear iTransformer
do 
for VE_len in 2 4 8 16 32 64 128
do 
for ParamRatio in 1 4
do 
    python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --H_order $H_order \
        --e_layers $e_layers \
        --c_out $enc_in \
        --d_model $d_model \
        --d_ff $d_model \
        --VE_len $VE_len \
        --ParamRatio $ParamRatio \
        --usePreprocess $usePreprocess \
        --des 'Exp' \
        --train_epochs $epoch \
        --patience $patience \
        --learning_rate $learning_rate \
        --lradj $lradj \
        --batch_size $batch_size \
        --itr $itr \
        --debug $debug
done
done
done
done

