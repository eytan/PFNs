from train import train_from_checkpoint
import argparse
import os

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--checkpoint_folder', type=str, default="/home/yl9959/mtpfn/ckpt/25-01-08_09-50-25__prior_mtgp__features_3__tasks_4__epochs_500__seqlen_200__attn_standard__task_onehot_linear__seed_0")
    argparser.add_argument('--pty', action='store_true', default=False)
    argparser.add_argument('--disable_wandb', action='store_true', default=False)
    argparser.add_argument('--epochs', type=int, default=None)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--lr', type=float, default=0.0001)
    argparser.add_argument('--batch_size', type=int, default=None)
    argparser.add_argument('--seq_len', type=int, default=None)
    argparser.add_argument('--steps_per_epoch', type=int, default=None)
    
    argparser.add_argument('--prior_type', type=str, default=None)
    argparser.add_argument('--eval_type', type=str, default=None)
    argparser.add_argument('--same_tasks_across_batch', action='store_true', default=None)
    # for gp-based
    argparser.add_argument('--lengthscale', type=float, default=None)
    # for unrelated
    argparser.add_argument('--uncorr_prob', type=float, default=None)
    # for mtgp-bias
    argparser.add_argument('--corr_init', type=float, default=None)
    
    if os.path.exists("/home/yl9959/mtpfn/ckpt"):
        output_dir = "/home/yl9959/mtpfn/ckpt"
        wandb_dir = "/home/yl9959/mtpfn/wandb_links"
    else:
        output_dir = "/home/lily_l/private_multitask_pfn/ckpt"
        wandb_dir = "/home/lily_l/private_multitask_pfn/wandb_links"
    argparser.add_argument('--output_dir', type=str, default=output_dir)
    argparser.add_argument('--wandb_dir', type=str, default=wandb_dir)
    argparser.add_argument('--save_val_results', action='store_true', default=False)
    
    
    args = argparser.parse_args()
    
    if args.disable_wandb:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
        
    train_from_checkpoint(**vars(args), wandb_mode=wandb_mode)
    