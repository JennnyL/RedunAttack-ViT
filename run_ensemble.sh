# attack
# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=global python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224 --batchsize 1
# eval
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224  --eval

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack vmifgsm --model swin_tiny_patch4_window7_224 --batchsize 16

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=600 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit dynamic 600 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit dynamic 400 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=200 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit dynamic 200 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=3000 ROBUST_TOKENS_TYPE=global python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
echo "pit global 3000 global tokens"  >> pit_log.txt
python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
echo "" >> pit_log.txt
echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=50 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
# echo "pit dynamic 50 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=20 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
# echo "pit dynamic 20 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=10 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
# echo "pit dynamic 10 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=1 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
# echo "pit dynamic 1 best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt


# HYPER_PARAM_TYPE=pit  NUM_ROBUST_TOKENS=1 ROBUST_TOKENS_TYPE=none python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
# echo "pit none best param"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt
