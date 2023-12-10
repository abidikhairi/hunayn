"""Investigate text dataset"""
import torch as th
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from hunayn.dataset.seq2seq import get_dataloaders


def main():
    """Main function
    """
    src_tknzr = PreTrainedTokenizerFast.from_pretrained('models/hunayn/protein_tokenizer')
    tgt_tknzr = PreTrainedTokenizerFast.from_pretrained('models/hunayn/english_tokenizer')

    df = pd.read_csv('data/10_12_23_dataset.csv', sep='\t')
    sequence_length = df['Sequence'].map(len)

    avg_sequence_length = sequence_length.mean()
    min_sequence_length = sequence_length.min()
    max_sequence_length = sequence_length.max()

    print(f'Average sequence length: {avg_sequence_length} residues')
    print(f'Minimum sequence length: {min_sequence_length} residues')
    print(f'Maximum sequence length: {max_sequence_length} residues')

    train_loader, _ = get_dataloaders("data/10_12_23_dataset.csv", src_tknzr, tgt_tknzr,
                                                 batch_size=16, num_workers=1, nheads=4)
    tokens = []

    for batch in tqdm(train_loader, total=len(train_loader)):
        tokens.append(batch['src'].shape[1] + batch['tgt'].shape[1])

    avg_tokens_per_batch = th.tensor(tokens, dtype=th.float).mean()

    print(f'Average Tokens per Batch: {avg_tokens_per_batch}')


if __name__ == "__main__":
    main()
