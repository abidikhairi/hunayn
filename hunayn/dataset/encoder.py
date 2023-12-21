"""Encoder dataset utilities"""
from copy import deepcopy
from functools import lru_cache
import random
import pandas as pd
import torch as th
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class HunaynEncoderDataset(Dataset):
    """
    Dataset class for training the Hunayn model.

    This dataset takes a tab-separated training file and a tokenizer for processing sequences.

    Args:
        training_file (str): Path to the training file in tab-separated format.
        tokenizer (PreTrainedTokenizerFast): Tokenizer for processing sequences.

    Attributes:
        df (pd.DataFrame): DataFrame containing the data loaded from the training file.
        tokenizer (PreTrainedTokenizerFast): Tokenizer used for processing sequences.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(index): Retrieves a specific sample from the dataset.

    """
    def __init__(self, training_file: str, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        Initializes a dataset for Hunayn model training.

        Args:
            training_file (str): Path to the training file in tab-separated format.
            tokenizer (PreTrainedTokenizerFast): Tokenizer for processing sequences.
        """
        super().__init__()

        self.df = pd.read_csv(training_file, sep='\t', header=None)
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    @lru_cache
    def token_ids(self):
        """Retrieve and cache token IDs from the tokenizer's vocabulary."""
        return list(self.tokenizer.vocab.values())

    def sample(self, tokens: th.Tensor, negative_idx: th.Tensor) -> th.Tensor:
        """
        Replace tokens at specified indices with randomly sampled tokens from the vocabulary.

        Args:
            tokens (th.Tensor): Tensor of token indices.
            negative_idx (th.Tensor): Indices at which to replace tokens.

        Returns:
            th.Tensor: Tensor with replaced tokens.
        """
        token_ids = self.token_ids()
        new_tokens = deepcopy(tokens)

        for idx in negative_idx:
            idx = idx.item()
            src = tokens[idx]
            token = random.choice(token_ids)

            if token == src:
                continue

            new_tokens[idx] = token

        return new_tokens

    def __getitem__(self, index):
        """
        Retrieves a specific sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[th.Tensor, th.Tensor]: Tuple containing the tokenized input sequence and corresponding labels.
        """
        if th.is_tensor(index):
            index = index.tolist()

        sequence = self.df.iloc[index, 0]
        tokens = th.tensor(self.tokenizer.encode(sequence), dtype=th.long)

        sample_size = int(len(tokens) * 0.15)
        randperm = th.randperm(len(tokens))
        negative_idx = randperm[:sample_size]

        tokens = self.sample(tokens, negative_idx)
        labels = th.ones_like(tokens)

        labels[negative_idx] = 0

        return tokens, labels


def collate_fn(items, tokenizer, nheads):
    """
    Collates a list of items into a batch for training or evaluation.

    Args:
        items (List[Tuple[th.Tensor, th.Tensor]]): List of tuples containing input sequences and labels.
        tokenizer: The tokenizer used for processing sequences.
        nheads: The number of encoder heads.

    Returns:
        Dict[str, th.Tensor]: A dictionary containing the tokenized input sequences ("x"),
            corresponding labels ("y"), and a mask indicating the positions of padding tokens ("mask").
    """
    xs = []
    ys = []

    for x, y in items:
        xs.append(x)
        ys.append(y)

    xs = pad_sequence(xs, batch_first=True,
                      padding_value=tokenizer.pad_token_id)
    ys = pad_sequence(ys, batch_first=True, padding_value=-100)
    label_mask = (ys != -100)

    src_mask = []
    for x in xs:
        mask = th.zeros((x.size(0), x.size(0))).bool()
        indices = th.argwhere(x == tokenizer.pad_token_id).flatten()
        mask[indices] = True
        src_mask.append(mask)

    src_mask = th.stack(src_mask, dim=0) \
            .repeat(nheads, 1, 1) \
            .float() # src_mask shape: (N x nheads, seq_len, seq_len)

    return {
        "x": xs,
        "y": ys,
        "mask": src_mask,
        "y_mask": label_mask
    }


def create_dataloader(sequence_file: str, tokenizer: PreTrainedTokenizerFast, nheads: int, 
                      batch_size: int = 64, num_workers: int = 4):
    """
    Creates a PyTorch DataLoader for training or evaluation.

    Args:
        sequence_file (str): Path to the file containing sequences and annotations.
        tokenizer (PreTrainedTokenizerFast): Tokenizer for processing sequences.
        nheads (int): The number of encoder heads.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.

    Returns:
        DataLoader: PyTorch DataLoader for the specified dataset.
    """
    dataset = HunaynEncoderDataset(
        training_file=sequence_file, tokenizer=tokenizer)

    def collate_fn_wrapper(x): return collate_fn(x, tokenizer, nheads)

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper)
