"""Seq2Seq datasets"""
import pandas as pd
import torch as th
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizerFast


class Seq2Seq(Dataset):
    """
    Dataset class for sequence-to-sequence tasks.

    Args:
        sequence_function_file (str): Path to the file containing sequence-function pairs.
        src_tknzr (PreTrainedTokenizerFast): Tokenizer for the source sequences.
        tgt_tknzr (PreTrainedTokenizerFast): Tokenizer for the target sequences.

    Attributes:
        src_tknzr (PreTrainedTokenizerFast): Tokenizer for the source sequences.
        tgt_tknzr (PreTrainedTokenizerFast): Tokenizer for the target sequences.
        frame (pd.DataFrame): DataFrame containing the sequence-function pairs.

    Example:
        ```python
        sequence_file = "path/to/sequence_function_pairs.tsv"
        src_tokenizer = PreTrainedTokenizerFast.from_pretrained("source_tokenizer")
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained("target_tokenizer")
        dataset = Seq2Seq(sequence_function_file=sequence_file, src_tknzr=src_tokenizer, tgt_tknzr=tgt_tokenizer)
        ```

    """
    def __init__(self, sequence_function_file: str, src_tknzr: PreTrainedTokenizerFast, tgt_tknzr: PreTrainedTokenizerFast) -> None:
        super().__init__()
        self.src_tknzr = src_tknzr
        self.tgt_tknzr = tgt_tknzr

        self.frame = pd.read_csv(sequence_function_file, sep='\t', header=0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        """
        Get a single item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[List[int], List[int], List[int]]: Tuple containing source sequence, target sequence, and labels.

        Example:
            ```python
            src, tgt, labels = dataset[0]
            ```
        """
        if th.is_tensor(index):
            index = index.tolist()
        sequence = self.frame.iloc[index, 0]
        annotation = self.frame.iloc[index, 1]

        src = self.src_tknzr(sequence)['input_ids']
        annotation = self.tgt_tknzr(annotation)['input_ids']
        tgt = annotation[:-1]
        labels = annotation[1:]

        return src, tgt, labels


def collate_fn(examples, src_tknzr: PreTrainedTokenizerFast, tgt_tknzr: PreTrainedTokenizerFast, nheads: int):
    """
    Collate function for sequence-to-sequence datasets.

    Args:
        examples (List[Tuple[List[int], List[int], List[int]]]): List of tuples containing source sequences, target sequences, and labels.
        src_tknzr (PreTrainedTokenizerFast): Tokenizer for the source sequences.
        tgt_tknzr (PreTrainedTokenizerFast): Tokenizer for the target sequences.
        nheads (int): Number of attention heads.

    Returns:
        Dict[str, th.Tensor]: Dictionary containing collated batches of source sequences, target sequences, source masks, target masks, and labels.

    Example:
        ```python
        batch = collate_fn([(src1, tgt1, labels1), (src2, tgt2, labels2), ...], src_tknzr, tgt_tknzr)
        ```
    """
    src = []
    tgt = []
    labels = []

    pad_token_id = src_tknzr.pad_token_id
    tgt_pad_token_id = tgt_tknzr.pad_token_id

    for s, t, y in examples:
        src.append(th.tensor(s).long())
        tgt.append(th.tensor(t).long())
        labels.append(th.tensor(y).long())

    src = pad_sequence(src, batch_first=True, padding_value=pad_token_id)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    if src.shape[1] > tgt.shape[1]:
        padding_rows = src.size(1) - tgt.size(1)
        padding_cols = src.size(0) - tgt.size(0)

        tgt = F.pad(tgt, (0, padding_rows, 0, padding_cols), value=tgt_pad_token_id)
        labels = F.pad(labels, (0, padding_rows, 0, padding_cols), value=-100)
    elif tgt.shape[1] > src.shape[1]:
        padding_rows = tgt.size(1) - src.size(1)
        padding_cols = tgt.size(0) - src.size(0)

        src = F.pad(src, (0, padding_rows, 0, padding_cols),
                    value=pad_token_id)

    src_mask = []
    for x in src:
        mask = th.zeros((x.size(0), x.size(0))).bool()
        indices = th.argwhere(x == pad_token_id).flatten()
        mask[indices] = True
        src_mask.append(mask)

    src_mask = th.stack(src_mask, dim=0) \
            .repeat(nheads, 1, 1) \
            .float() # src_mask shape: (N x nheads, seq_len, seq_len)

    tgt_mask = []
    for y in tgt:
        attn_shape = (y.size(0), y.size(0))
        mask = th.tril(th.ones(attn_shape)).bool()
        indices = th.argwhere(y == tgt_pad_token_id).flatten()
        mask = ~mask
        mask[indices] = True
        tgt_mask.append(mask)

    tgt_mask = th.stack(tgt_mask, dim=0) \
            .repeat(nheads, 1, 1) \
            .float() # tgt_mask shape: (N x nheads, seq_len, seq_len)

    return {
        "src": src,
        "tgt": tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "labels": labels
    }

def get_test_dataloader(protein_function_file: str, src_tknzr: PreTrainedTokenizerFast, tgt_tknzr: PreTrainedTokenizerFast,
                        batch_size: int = 4, num_workers: int = 2, nheads: int = 4) -> DataLoader:
    """
    Get a DataLoader for the test dataset.

    Args:
        protein_function_file (str): Path to the protein function file.
        src_tknzr (PreTrainedTokenizerFast): Source tokenizer for sequences.
        tgt_tknzr (PreTrainedTokenizerFast): Target tokenizer for function annotations.
        batch_size (int, optional): Batch size. Defaults to 4.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 2.
        nheads (int, optional): Number of attention heads. Defaults to 4.

    Returns:
        DataLoader: DataLoader for the test dataset.

    Example:
        ```python
        # Example usage:
        # test_dataloader = get_test_dataloader("test_data.tsv", src_tokenizer, tgt_tokenizer, batch_size=8)
        # for batch in test_dataloader:
        #     # Process the batch
        # ```
    """
    dataset = Seq2Seq(protein_function_file, src_tknzr, tgt_tknzr)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              collate_fn=lambda examples: collate_fn(examples, src_tknzr, tgt_tknzr, nheads))

    return data_loader

def get_dataloaders(protein_function_file: str, src_tknzr: PreTrainedTokenizerFast, tgt_tknzr: PreTrainedTokenizerFast,
                    train_size: float = 0.8, batch_size: int = 4, num_workers: int = 2, nheads: int = 4):
    """
    Get DataLoader instances for training and validation from a protein function dataset.

    Args:
        protein_function_file (str): Path to the file containing protein function information.
        src_tknzr (PreTrainedTokenizerFast): Tokenizer for the source sequences.
        tgt_tknzr (PreTrainedTokenizerFast): Tokenizer for the target sequences.
        train_size (float, optional): Percentage of data to use for training. Defaults to 0.8.
        batch_size (int, optional): Batch size. Defaults to 4.
        num_workers (int, optional): Number of workers for data loading. Defaults to 2.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing DataLoader instances for training and validation.

    Example:
        ```python
        src_tokenizer = PreTrainedTokenizerFast.from_pretrained("source_tokenizer")
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained("target_tokenizer")
        train_loader, valid_loader = get_dataloaders("path/to/protein_function_file.tsv", src_tokenizer, tgt_tokenizer)
        ```
    """
    dataset = Seq2Seq(protein_function_file, src_tknzr, tgt_tknzr)
    num_samples = len(dataset)
    num_train_samples = int(num_samples * train_size)
    num_valid_samples = num_samples - num_train_samples

    trainset, validset = random_split(dataset, [num_train_samples, num_valid_samples])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=lambda examples: collate_fn(examples, src_tknzr, tgt_tknzr, nheads))
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              collate_fn=lambda examples: collate_fn(examples, src_tknzr, tgt_tknzr, nheads))

    return (train_loader, valid_loader)
