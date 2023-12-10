"""Explode protein function column"""
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize


INPUT_FILE = "data/uniprot/processed_02.tsv"
OUTPUT_FILE = "data/uniprot/processed_03.tsv"


def main():
    """
    Main function to read a CSV file, tokenize function sentences, and create a new dataset.

    Example:
        ```python
        # Example usage:
        # INPUT_FILE = "data/uniprot/processed_01.tsv"
        # OUTPUT_FILE = "data/uniprot/processed_03.tsv"
        # main()
        ```

    Note:
        This function reads a CSV file at `INPUT_FILE`, tokenizes sentences in the 'Function' column,
        and creates a new dataset with columns 'Sequence' and 'FUNCTION'. The processed data is saved
        to the file specified by `OUTPUT_FILE`.
    """
    df = pd.read_csv(INPUT_FILE, sep='\t', header=0)

    dataset = {
        "Sequence": [],
        "FUNCTION": []
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        functions = list(filter(lambda x: len(x) > 1, sent_tokenize(row.Function)))
        sequence = row.Sequence

        for f in functions:
            dataset['Sequence'].append(sequence)
            dataset["FUNCTION"].append(f)

    pd.DataFrame(dataset).to_csv(OUTPUT_FILE, sep='\t', index=False)


if __name__ == '__main__':
    main()
