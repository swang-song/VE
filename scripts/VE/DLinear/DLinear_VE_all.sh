export CUDA_VISIBLE_DEVICES=0


usePreprocess=softmax

debug=0
epoch=10
itr=3
patience=3

model_name=DLinear








root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=ECL
data_name=custom
enc_in=321
learning_rate=0.001
lradj=type1
batch_size=16
seq_len=336
for pred_len in 96 192
do 
for VE_len in 2 4 8 16 32 64 128
do 
for ParamRatio in 0.25 1 4
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








root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
enc_in=7
learning_rate=0.005
lradj=type1
batch_size=32
seq_len=336
for pred_len in 96 192
do 
for VE_len in 2 4 8 16 32 64 128
do 
for ParamRatio in 0.25 1 4
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








root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
enc_in=7
learning_rate=0.005
lradj=type1
batch_size=32
seq_len=336
for pred_len in 96 192
do 
for VE_len in 2 4 8 16 32 64 128
do 
for ParamRatio in 0.25 1 4
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








root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
enc_in=21
learning_rate=0.0001
lradj=type1
batch_size=16
seq_len=336
for pred_len in 96 192
do 
for VE_len in 2 4 8 16 32 64 128
do 
for ParamRatio in 0.25 1 4
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


