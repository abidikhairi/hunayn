"""Build the testing dataset"""
import re
import pandas as pd
from tqdm import tqdm
from nltk import sent_tokenize



INPUT_FILE = "data/uniprot/raw_testset.tsv"
OUTPUT_FILE = "data/testset.tsv"


def normalize_function_column(text: str) -> str:
    """
    Normalize the text in the 'Function' column by converting it to lowercase.

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text in lowercase.
    """
    return text.lower()

def remove_references(text: str) -> str:
    """
    Remove references from the input text.

    Args:
        text (str): The input text containing references.

    Returns:
        str: The text with references removed.
    """
    # Remove references in the format (pubmed:[0-9]+(?:, pubmed:[0-9]+)*)
    res1 = re.sub(r"\(pubmed:[0-9]+(?:, pubmed:[0-9]+)*\)", "", text)

    # Remove references in the format {eco:[0-9]+|pubmed:[0-9]+,eco:[0-9]+|uniprot_acc:[a-z0-9]+}
    return re.sub(r"\{.*\}", "", res1)

def remove_function_keyword(text: str) -> str:
    """
    Remove the 'function:' keyword from the input text.

    Args:
        text (str): The input text containing the 'function:' keyword.

    Returns:
        str: The text with the 'function:' keyword removed.
    """
    return text.replace("function: ", '')

def remove_eg_text(text: str) -> str:
    """
    Removes example (e.g.) text from the input string.

    Args:
        text (str): The input text containing examples.

    Returns:
        str: The text with example content removed.

    Example:
        ```python
        # Example usage:
        # input_text = "This is a sentence (e.g., with an example)."
        # result = remove_eg_text(input_text)
        # print(result)
        ```

    Note:
        This function uses a regular expression to find and remove text within parentheses
        containing "e.g." (e.g., "This is an example sentence (e.g., with additional text).").
    """
    return re.sub(r"\(.*e.g.*\)", "", text)

def main():
    df = pd.read_csv(INPUT_FILE, sep="\t", header=0)
    
    df.dropna(axis='index', inplace=True)
    
    df['Function'] = df['Function'].map(normalize_function_column) \
        .map(remove_references) \
        .map(remove_function_keyword) \
        .map(remove_eg_text)


    dataset = {
        "Sequence": [],
        "Function": [],
        "Organism": []
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):
        functions = list(filter(lambda x: len(x) > 1, sent_tokenize(row.Function)))
        sequence = row.Sequence
        organism = row.Organism

        for f in functions:
            dataset['Sequence'].append(sequence)
            dataset["Function"].append(f)
            dataset['Organism'].append(organism)

    pd.DataFrame(dataset).to_csv(OUTPUT_FILE, sep='\t', index=False)


if __name__ == '__main__':
    main()
