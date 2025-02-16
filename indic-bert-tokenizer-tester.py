#!/usr/bin/env python3
import sys

if len(sys.argv) < 2:
    print("requires <tokenizer-model>")
    sys.exit(0)

from indic_bert_tokenizer import IndicBertWordPieceTokenizer

tokenizer = IndicBertWordPieceTokenizer(sys.argv[1])

# read the lines from stdin
while (line := sys.stdin.readline()): #.strip() != "":
    #print(line)
    # strip the new line character
    line = line.strip()
    toks = tokenizer.encode(line)
    #print(toks.ids)
    print(toks.tokens)
    #decoded = tokenizer.decode(toks.ids)
    #print(decoded)
    decoded = tokenizer.decode(toks.ids)
    print(decoded)

