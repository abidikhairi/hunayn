"""Filter in protein with functions"""
import pandas as pd


INPUT_FILE = "data/uniprot/raw_dataset_01.tsv"
OUTPUT_FILE = "data/uniprot/processed_01.tsv"


def main():
    """
    Main function to process a CSV file, removing rows with NaN values and saving the result.

    Example:
        ```python
        # Example usage:
        # INPUT_FILE = "path/to/input_file.csv"
        # OUTPUT_FILE = "path/to/output_file.csv"
        # main(INPUT_FILE, OUTPUT_FILE)
        ```

    Note:
        Ensure that the CSV file at `INPUT_FILE` has the appropriate format, and the processed data
        will be saved to the file specified by `OUTPUT_FILE`.
    """
    df = pd.read_csv(INPUT_FILE, sep="\t", header=0)    
    df.dropna(axis='index').to_csv(OUTPUT_FILE, sep="\t", index=False)


if __name__ == "__main__":
    main()
