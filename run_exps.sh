# model_name=vit_base_patch16_224
model_name=pit_b_224

# for v in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
# do
#     ATTN_DROP_RATE=$v python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack sparse --model $model_name --batchsize 16
#     echo "" >> exp_results/pit_sparse.log
#     echo attn_drop_rate: >> exp_results/pit_sparse.log
#     echo $v >> exp_results/pit_sparse.log
#     python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack sparse --model $model_name --batchsize 16 --eval >> exp_results/pit_sparse.log
# done


# for v in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     SHUFFLE_PROB=$v python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack shuffle --model $model_name --batchsize 16
#     echo "" >> exp_results/pit_shuffle.log
#     echo shuffle_prob: >> exp_results/pit_shuffle.log
#     echo $v >> exp_results/pit_shuffle.log
#     python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack shuffle --model $model_name --batchsize 16 --eval >> exp_results/pit_shuffle.log
# done


# for v in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     REST_P=$v python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack rest --model $model_name --batchsize 16
#     echo "" >> exp_results/pit_rest.log
#     echo rest_p: >> exp_results/pit_rest.log
#     echo $v >> exp_results/pit_rest.log
#     python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack rest --model $model_name --batchsize 16 --eval >> exp_results/pit_rest.log
# done


# for n in 1 2 3 4 5
# do
#     for p in 0.1 0.2 0.3 0.4 0.5
#     do
#         MOE_N=$n MOE_PROB=$p python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack moe --model $model_name --batchsize 16

#         echo "" >> exp_results/pit_moe.log
#         echo moe_n: >> exp_results/pit_moe.log
#         echo $n >> exp_results/pit_moe.log

#         echo moe_p: >> exp_results/pit_moe.log
#         echo $p >> exp_results/pit_moe.log

#         python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack moe --model $model_name --batchsize 16 --eval >> exp_results/pit_moe.log
#     done
# done


for n in 0.25
do
    for p in 0.7 0.8 0.9 1.0 # 0.1 0.2 0.3 0.4 0.5
    do
        SHUFFLE_PROB=$n SHUFFLE_RATIO=$p python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ishuffle --model $model_name --batchsize 16

        echo "" >> exp_results/pit_ishuffle.log
        echo shuffle_prob: >> exp_results/pit_ishuffle.log
        echo $n >> exp_results/pit_ishuffle.log

        echo shuffle_prob: >> exp_results/pit_ishuffle.log
        echo $p >> exp_results/pit_ishuffle.log

        python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack moe --model $model_name --batchsize 16 --eval >> exp_results/pit_ishuffle.log
    done
done



for n in 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0
do
    for p in 0.6 0.7 0.8 0.9 1.0 # 0.1 0.2 0.3 0.4 0.5
    do
        SHUFFLE_PROB=$n SHUFFLE_RATIO=$p python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ishuffle --model $model_name --batchsize 16

        echo "" >> exp_results/pit_ishuffle.log
        echo shuffle_prob: >> exp_results/pit_ishuffle.log
        echo $n >> exp_results/pit_ishuffle.log

        echo shuffle_prob: >> exp_results/pit_ishuffle.log
        echo $p >> exp_results/pit_ishuffle.log

        python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack moe --model $model_name --batchsize 16 --eval >> exp_results/pit_ishuffle.log
    done
done


