## Hunayn training notes

#### Tokenization:
Following ProGPT2, the tokenization is done using the BPE [ref] algorithm on all the reviewed Human protein sequences (a total of 20435 sequence). In addition, the standard 20 amino acids were added to the tokenizer.

What's a token $\implies$ an Oligomer.

**Oligomers**: An oligomer is a molecule that consists of a few monomer (amino acids) units.

#### TODO
- Train tokenizer on the TrEMBL database (250 Million sequences).