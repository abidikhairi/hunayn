"""Main file for training Hunayn"""
import torch as th
from pytorch_lightning import Trainer, loggers, callbacks
from transformers import PreTrainedTokenizerFast

from hunayn.utils import get_default_parser
from hunayn.config import TransformerConfig, OptimizerConfig
from hunayn.dataset import get_dataloaders
from hunayn.model import HunaynTrainer


def main(args):
    """
    Main function to train the Hunayn model.

    Args:
        args (Namespace): Command-line arguments.

    Example:
        ```python
        # Example command-line usage:
        # python train_hunayn.py --d_model 512 --d_ff 256 --nhead 4 --dropout 0.5 --num_encoder_layers 6 --num_decoder_layers 6
        #                         --batch_size 16 --num_workers 4 --learning_rate 0.001 --beta1 0.9 --beta2 0.988 --warmup_steps 2000
        #                         --src_tokenizer_path "path/to/src_tokenizer" --tgt_tokenizer_path "path/to/tgt_tokenizer"
        #                         --input_file "path/to/input_file.csv"
        ```

    Note:
        Ensure that the tokenizers at `src_tokenizer_path` and `tgt_tokenizer_path` are compatible with
        the PreTrainedTokenizerFast class from the transformers library.
    """
    d_model = args.d_model
    d_ff = args.d_ff
    nhead = args.nhead
    dropout = args.dropout
    num_encoder_layers = args.num_encoder_layers
    num_decoder_layers = args.num_decoder_layers
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    warmup_steps = args.warmup_steps
    src_tokenizer_path = args.src_tokenizer_path
    tgt_tokenizer_path = args.tgt_tokenizer_path
    input_file = args.input_file

    src_tknzr = PreTrainedTokenizerFast.from_pretrained(src_tokenizer_path)
    tgt_tknzr = PreTrainedTokenizerFast.from_pretrained(tgt_tokenizer_path)

    src_vocab_size = src_tknzr.vocab_size
    src_padding_idx = src_tknzr.pad_token_id
    tgt_vocab_size = tgt_tknzr.vocab_size

    train_loader, valid_loader = get_dataloaders(input_file, src_tknzr, tgt_tknzr,
                                                 batch_size=batch_size, num_workers=num_workers, nheads=nhead)

    model_config = TransformerConfig(d_model=d_model, d_ff=d_ff, nhead=nhead, dropout=dropout,
                                     num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers,
                                     src_vocab_size=src_vocab_size, src_padding_idx=src_padding_idx,
                                     tgt_vocab_size=tgt_vocab_size, tgt_padding_idx=-100)

    optim_config = OptimizerConfig(learning_rate=learning_rate, beta1=beta1, beta2=beta2,
                                   warmup_steps=warmup_steps, d_model=d_model)

    model = HunaynTrainer(model_config, optim_config)

    csv_logger = loggers.CSVLogger('logs', name="hunayn", version=1)
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath="model/checkpoints", filename="hunayn_v1.ckpt", monitor="train/loss")

    trainer = Trainer(accelerator="gpu", log_every_n_steps=50, max_epochs=10,
                      logger=[csv_logger], callbacks=[model_checkpoint_callback], enable_checkpointing=True)

    trainer.fit(model, train_loader, valid_loader)

    th.save(model.state_dict(), 'model/hunayn/weights_v1.pth')

if __name__ == '__main__':
    parser = get_default_parser()

    args = parser.parse_args()

    main(args)
