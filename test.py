"""Main file for training Hunayn"""
import argparse
from pytorch_lightning import Trainer, loggers
from transformers import PreTrainedTokenizerFast

from hunayn.dataset import get_test_dataloader
from hunayn.model import HunaynTrainer


def main(args):
    """
    Execute the main testing routine.

    This function loads the necessary tokenizers, the pre-trained model from a specified checkpoint path,
    creates a test data loader, and runs the testing using a Lightning Trainer.

    Args:
        args (Namespace): Command-line arguments containing the path to the checkpoint.
    Example:
        ```python
        # Example command-line usage:
        # python test.py --checkpoint-path models/hunayn/checkpoints/hunayn_v1.ckpt
        ```
    Returns:
        None
    """
    src_tknzr = PreTrainedTokenizerFast.from_pretrained(
        "models/hunayn/protein_tokenizer")
    tgt_tknzr = PreTrainedTokenizerFast.from_pretrained(
        "models/hunayn/english_tokenizer")

    model = HunaynTrainer.load_from_checkpoint(args.checkpoint_path)

    nheads = model.model_config.nhead

    test_loader = get_test_dataloader(
        'data/testset.tsv', src_tknzr, tgt_tknzr, batch_size=2, num_workers=4, nheads=nheads)

    csv_logger = loggers.CSVLogger(save_dir="logs", name="tests", version=1)

    trainer = Trainer(accelerator='gpu', logger=[csv_logger])

    trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-path', required=True, type=str, help="Model checkpoint path")

    main(parser.parse_args())
