# attack
# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=global python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224 --batchsize 1
# eval
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224  --eval

# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack vmifgsm --model swin_tiny_patch4_window7_224 --batchsize 16

NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=none python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 8
echo "pit none"  >> pit_log.txt
python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
echo "" >> pit_log.txt
echo "" >> pit_log.txt

# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 400 dynamic"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# NUM_ROBUST_TOKENS=200 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 200 dynamic"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt


# NUM_ROBUST_TOKENS=100 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 100 dynamic"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt


# NUM_ROBUST_TOKENS=50 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 50 ROBUST_TOKENS_TYPE"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt


# NUM_ROBUST_TOKENS=600 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 600 dynamic"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# NUM_ROBUST_TOKENS=20 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 20 ROBUST_TOKENS_TYPE"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt

# NUM_ROBUST_TOKENS=10 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 1
# echo "pit 10 ROBUST_TOKENS_TYPE"  >> pit_log.txt
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224  --eval  >> pit_log.txt
# echo "" >> pit_log.txt
# echo "" >> pit_log.txt


# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=global python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 2

# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack vmifgsm --model vit_base_patch16_224  --eval