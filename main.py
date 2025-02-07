from typing import Optional, Union
# Description: This file is the main entry point for the application.
import argparse, warnings
from util.log_util import Log
from src.t5_finetune import finetune_t5
from src.t5_train_model import t5_training_process
from src.fasttext_finetune import finetune_fasttext
from src.fasttext_train_model import fasttext_training_process
warnings.filterwarnings("ignore")
def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--choose_finetune', type=int, default=0, help='choose finetune or not')
    args.add_argument('--method_type', type=str, default='t5', help='model type')
    args.add_argument('--model_type', type=str, default='view_determine_zh', help='model name')
    args.add_argument('--language', type=str, default='zh', help='language')
    
    # Training parameters
    args.add_argument('--device_ids', default=3, help='device ids')
    args.add_argument('--default_size', type=str, default='small', help='model size choose small or large')
    args.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    args.add_argument('--batch_size', type=int, default=16, help='batch size')
    args.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    return args.parse_args()
logger = Log('main')
args = arg_parser()
def run(args=args, logger=logger):
    logger.log_info("Starting the application.")
    
    logger.log_info("Parsing the arguments.")
    logger.log_info(f"Choose finetune: {args.choose_finetune}")
    logger.log_info(f"Method type: {args.method_type}")
    logger.log_info(f"Model type: {args.model_type}")
    logger.log_info(f"Language: {args.language}")
    logger.log_info(f"Device ids: {args.device_ids}")
    logger.log_info(f"Default size: {args.default_size}")
    logger.log_info(f"Number of epochs: {args.num_epochs}")
    logger.log_info(f"Batch size: {args.batch_size}")
    logger.log_info(f"Learning rate: {args.lr}")
    
    if args.choose_finetune==1:
        if args.method_type == 't5':
            logger.log_info("Finetuning T5 model.")
            finetune_t5(
                model_type=args.model_type, 
                language=args.language, 
                device_ids=args.device_ids,
                default_size=args.default_size,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                logger=logger
            )
        elif args.method_type == 'fasttext':
            logger.log_info("Finetuning FastText model.")
            finetune_fasttext(
                model_type=args.model_type, 
                logger=logger
            )
            
    else:
        if args.method_type == 't5':
            logger.log_info("Training T5 model.")
            t5_training_process(
                model_type=args.model_type, 
                language=args.language, 
                device_ids=args.device_ids,
                default_size=args.default_size,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                logger=logger
            )
        elif args.method_type == 'fasttext':
            logger.log_info("Training FastText model.")
            fasttext_training_process(
                model_type=args.model_type, 
                logger=logger
            )
if __name__ == '__main__':
    run()