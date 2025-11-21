import argparse
from pathlib import Path
import pandas as pd
from pprint import pprint

from configs.fine_tune_config import ft_cfg
from fine_tuning.preprocessing import prep_data


def parse_args():
    parser = argparse.ArgumentParser(description="Choose to train model or evaluate. If evaluating, must specify trained model dir")
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['train', 'eval'],
        default='train',
        help='Specify to train or evaluate a previously trained model. (Default is train)'
    )

    parser.add_argument(
        '-r', '--resume',
        action='store_true',
        help='Resume a previously unfinished training; references `--lora_dir` to load lora for resuming'
    )

    parser.add_argument(
        '-d', '--lora_dir',
        type=str,
        default=None,
        help='Specify the path to load the model for evaluation'
    )

    args = parser.parse_args()
    if (args.mode =='eval' or args.resume) and args.lora_dir is None:
        raise Exception("Error: must specify path to model for evaluation if in evaluation mode.")
    elif args.mode =='eval' and args.lora_dir is not None:
        try:
            args.lora_dir = Path(args.lora_dir).resolve()
        except Exception as e:
            print(f"Error: {args.lora_dir} is an invalid path\n{e}")

    return args

def main():
    args = parse_args()
    if args.mode == 'train':        
        from fine_tuning.initialize import tuner, tokenizer

        if args.resume:
            tuner.train(resume_from_checkpoint=args.lora_dir)        
        else:
            tuner.train()
            
        tuner.save_model(f'{ft_cfg.log_dir}/final')
        tokenizer.save_pretrained(f'{ft_cfg.log_dir}/final')

    elif args.mode == 'eval':
        from fine_tuning.utils import load_model
        from fine_tuning.eval import evaluate_split, analyze_prompt_lengths
        
        model, tokenizer = load_model(args.lora_dir, device=ft_cfg.device)
        train_df, test_df = pd.read_csv(ft_cfg.train_csv), pd.read_csv(ft_cfg.test_csv)
        test_ds, _ = prep_data(tokenizer, ft_cfg, mode='eval')
        train_ds, _ = prep_data(tokenizer, ft_cfg, mode='train')
        pprint(f"Train data prompt token length statistics: {analyze_prompt_lengths(train_df, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, max_length=610)}")
        pprint(f"Test data prompt token length statistics: {analyze_prompt_lengths(test_df, tokenizer, ft_cfg.x_col_name, ft_cfg.y_col_name, max_length=610)}")
        
        pprint(f"***TEST SAMPLE:\n{test_ds[0]}\n\n***TRAIN SAMPLE:\n{train_ds[0]}")

        evaluate_split(
            model,
            tokenizer,
            ft_cfg.ctx,
            test_ds,
            'text',
            'labels',
            ft_cfg.device,
            view_n_model_predictions=10
        )


if __name__ == '__main__':
    main()