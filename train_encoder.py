"""Main file for training Hunayn encoder"""
import argparse
from datetime import datetime
import torch as th
from pytorch_lightning import Trainer, loggers as pl_loggers
from transformers import PreTrainedTokenizerFast

from hunayn.utils import lightning_callbacks, lightning_loggers
from hunayn.config import OptimizerConfig, EncoderConfig
from hunayn.dataset import create_encoder_dataloader
from hunayn.model import HunaynEncoder


def main(args):
    """
    Main function to train the Hunayn encoder model.

    Args:
        args (Namespace): Command-line arguments.
    """
    d_model = args.d_model
    d_ff = args.d_ff
    nhead = args.nhead
    dropout = args.dropout
    num_encoder_layers = args.num_encoder_layers
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    src_tokenizer_path = args.src_tokenizer_path
    train_file = args.train_file
    valid_file = args.valid_file

    src_tknzr = PreTrainedTokenizerFast.from_pretrained(src_tokenizer_path)

    src_vocab_size = src_tknzr.vocab_size
    src_padding_idx = src_tknzr.pad_token_id

    train_loader = create_encoder_dataloader(train_file, src_tknzr, nhead, batch_size, num_workers)
    valid_loader = create_encoder_dataloader(valid_file, src_tknzr, nhead, batch_size, num_workers)

    model_config = EncoderConfig(d_model=d_model, d_ff=d_ff, nhead=nhead, dropout=dropout,
                                     num_encoder_layers=num_encoder_layers,
                                     src_vocab_size=src_vocab_size, src_padding_idx=src_padding_idx)

    optim_config = OptimizerConfig(learning_rate=learning_rate, beta1=beta1, beta2=beta2, d_model=d_model)

    model = HunaynEncoder(model_config, optim_config)

    loggers = lightning_loggers()
    callbacks = lightning_callbacks()

    now = datetime.now().strftime("%Y%m%d")
    loggers.append(pl_loggers.WandbLogger(project="hunayn", name=f"encoder-experim-{now}"))

    trainer = Trainer(accelerator="gpu", log_every_n_steps=50, max_epochs=50,
                      logger=loggers, callbacks=callbacks, enable_checkpointing=True)

    trainer.fit(model, train_loader, valid_loader)

    th.save(model.encoder.state_dict(), 'models/hunayn/encoder_weights_v1.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--d_model', type=int, required=True, help='Dimensionality of the model')
    parser.add_argument('--d_ff', type=int, required=True, help='Dimensionality of the feed-forward networks')
    parser.add_argument('--nhead', type=int, required=True, help='Number of heads in multi-head attention')
    parser.add_argument('--dropout', type=float, required=True, help='Dropout probability')
    parser.add_argument('--num_encoder_layers', type=int, required=True, help='Number of encoder layers')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, required=True, help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--beta1', type=float, required=True, help='Beta1 value for Adam optimizer')
    parser.add_argument('--beta2', type=float, required=True, help='Beta2 value for Adam optimizer')
    parser.add_argument('--src_tokenizer_path', type=str, required=True, help='Path to the source tokenizer')
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training file')
    parser.add_argument('--valid_file', type=str, required=True, help='Path to the validation file')

    args = parser.parse_args()

    main(args)
