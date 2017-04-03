The system requires three kinds of input:
1. SRL data
2. Lexicon
3. VSM Lookup

1. SRL Data
The default format for SRL data uses two types of files: sentence files and annotation files.
*Sentence files* are tab-separated one sentence per line with POS tags, lemmas and dependency relations.
The format is similar to CoNLL-2009 or MaltTab with all columns being merged in a single line.

[# tokens][tokens][POS tags][dependency labels][dependency heads][O][lemmas]

*Frame element files* are tab separated with the following column semantics:

[optional]
[optional]
[# of roles]
[frame name]
[lemma.pos]
[position of the FEE in the sentence]
[FEE string]
[line# in the sentence file (incl. 0)]
[role1]
[position1]
[role2]
[position2]
etc.

2. Lexicon data
Lexicon files are simple lists of frames and predicates that can evoke them, tab-separated, one pair per line.

3. VSM data
We use the standard word embeddings format, where each line corresponds to a word followed by its vector representation.