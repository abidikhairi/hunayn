"""Train a tokenizer"""
import argparse
from tokenizers import Tokenizer, processors, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def text_file_iterator(text_file_path: str):
    """
    Iterator function for reading lines from a text file.

    Args:
        text_file_path (str): Path to the text file.

    Yields:
        str: Stripped lines from the text file.

    Example:
        ```python
        file_path = 'path/to/your/text/file.txt'
        for line in text_file_iterator(file_path):
            print(line)
        ```

    """
    with open(text_file_path, 'r', encoding='utf-8') as stream:
        for line in stream:
            yield line.strip()


def main(save_path: str, text_file: str):
    """
    Creates a Protein FastTokenizer compatible with Hugging Face's transformers library and saves it.

    Args:
        save_path (str): The path where the trained tokenizer will be saved.
        text_file (str): The path to the text file that contains protein sequences.

    This script sets up a Protein FastTokenizer using the `tokenizers` library, adds special tokens,
    and saves the trained tokenizer using the `PreTrainedTokenizerFast` from Hugging Face's transformers library.
    The resulting tokenizer is compatible with the Hugging Face's transformers library.

    Usage:
        python train_tokenizer.py --text-file /path/to/file.txt --save-path /path/to/save/tokenizer

    Args:
        --save-path (str): The path where the trained tokenizer will be saved.
        --text-file (str): The path to the text file that contains protein sequences.
    """
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    bos_token = '<bos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    unk_token = '<unk>'

    tokenizer = Tokenizer(models.BPE(unk_token=unk_token))

    special_tokens = [bos_token, eos_token, pad_token, unk_token]

    tokenizer.add_special_tokens(special_tokens)

    sos_token_id = tokenizer.token_to_id(bos_token)
    eos_token_id = tokenizer.token_to_id(eos_token)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos>:0 $A:0 <eos>:0",
        special_tokens=[('<bos>', sos_token_id), ('<eos>', eos_token_id)]
    )

    trainer = trainers.BpeTrainer(
        initial_alphabet=amino_acids, show_progress=True, vocab_size=50000
    )

    tokenizer.train_from_iterator(text_file_iterator(
        text_file), trainer=trainer, length=20435)

    trained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token)

    trained_tokenizer.save_pretrained(save_path)

    loaded_tknzr = PreTrainedTokenizerFast.from_pretrained(save_path)
    print(loaded_tknzr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Creates a Protein FastTokenizer compatible with Huggingfaces")

    parser.add_argument('--text-file', type=str,
                        required=True, help="Sequences file path (each line is a sequence).")
    parser.add_argument('--save-path', type=str,
                        required=True, help="tokenizer save path")

    args = parser.parse_args()

    main(args.save_path, args.text_file)
