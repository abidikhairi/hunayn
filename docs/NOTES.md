## Hunayn training notes

#### Tokenization:
Following ProGPT2, the tokenization is done using the Byte-Pair Encoding algorithm on all the reviewed Human protein sequences (a total of 20435 sequence). In addition, the standard 20 amino acids were added to the tokenizer.

What's a token $\implies$ an Oligomer.

**Oligomers**: An oligomer is a molecule that consists of a few monomer (amino acids) units.

#### TODO
- Train tokenizer on the TrEMBL database (250 Million sequences).

### Updates:
- 10-12-2023:
    - Text processed to reduce input size (protein and function wise)
    - Each protein function is splitted into sentences, then, a pair of (protein, function) is constructed
    by repeating the target protein with its corresponding functions.
    - New BPE tokenizer is trained on the new downloaded dataset.
    - New Dataset is: **(length:[1 to 100]) AND (existence:1) AND (reviewed:true)**
