export CUDA_VISIBLE_DEVICES=0


usePreprocess=softmax

debug=0
epoch=50
itr=3
patience=3

model_name=FITS








root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=ECL
data_name=custom
enc_in=321
H_order=10
base_T=24
learning_rate=0.0005
lradj=FITS_type3
batch_size=128
seq_len=720
for pred_len in 96 192
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
        --base_T $base_T \
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
H_order=6
base_T=24
learning_rate=0.0005
lradj=FITS_type3
batch_size=128
seq_len=360
for pred_len in 96 192
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
        --base_T $base_T \
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
H_order=6
base_T=24
learning_rate=0.0005
lradj=FITS_type3
batch_size=128
seq_len=720
for pred_len in 96 192
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
        --base_T $base_T \
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
H_order=12
base_T=144
learning_rate=0.0005
lradj=FITS_type3
batch_size=64
seq_len=720
for pred_len in 96 192
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
        --base_T $base_T \
        --VE_len $VE_len \
        --ParamRatio $ParamRatio \
        --usePreprocess none \
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


