"""Main file for training Hunayn"""
from pytorch_lightning import Trainer
from transformers import PreTrainedTokenizerFast

from hunayn.dataset import get_test_dataloader
from hunayn.model import HunaynTrainer


def main():
    """
    Execute the main testing routine.

    This function loads the necessary tokenizers and the pre-trained model, creates a test data loader,
    and runs the testing using a Lightning Trainer.

    Args:
        None

    Returns:
        None
    """
    src_tknzr = PreTrainedTokenizerFast.from_pretrained(
        "models/hunayn/protein_tokenizer")
    tgt_tknzr = PreTrainedTokenizerFast.from_pretrained(
        "models/hunayn/english_tokenizer")

    model = HunaynTrainer.load_from_checkpoint(
        'models/hunayn/checkpoints/hunayn_v1.ckpt')

    nheads = model.model_config.nhead

    test_loader = get_test_dataloader(
        'data/testset.tsv', src_tknzr, tgt_tknzr, batch_size=2, num_workers=4, nheads=nheads)

    trainer = Trainer(accelerator='gpu')

    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
