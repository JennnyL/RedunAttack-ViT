# # attack
# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224 --batchsize 16
# # eval
# python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model vit_base_patch16_224  --eval

NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model swin_tiny_patch4_window7_224 --batchsize 16


# NUM_ROBUST_TOKENS=400 ROBUST_TOKENS_TYPE=dynamic python main.py --input_dir /home/cxu-serve/p62/zzh136/global_prune/attack/data/TransferAttack/data  --output_dir adv_data --attack ll2s --model pit_b_224 --batchsize 16
