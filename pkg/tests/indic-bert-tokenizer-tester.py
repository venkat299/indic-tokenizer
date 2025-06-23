#!/usr/bin/env python3

# ensure the package is installed.
# pip install -i https://test.pypi.org/simple/ indic-tokenizer -U

import sys

if len(sys.argv) < 2:
    print("requires <tokenizer-model>")
    sys.exit(0)

from indic_tokenizer import IndicBertWordPieceTokenizer

tokenizer = IndicBertWordPieceTokenizer(sys.argv[1])

# read the lines from stdin
while (line := sys.stdin.readline()): #.strip() != "":
    # strip the new line character
    line = line.strip()
    toks = tokenizer.encode(line)
    # get the token ids.
    print(toks.ids)
    # print the tokens (but encoded in singular unicode)
    print(toks.tokens)
    # print the decoded tokens
    print([tokenizer.decode_string(tok) for tok in toks.tokens])
    # print the decoded string
    decoded = tokenizer.decode(toks.ids)
    print(decoded)

