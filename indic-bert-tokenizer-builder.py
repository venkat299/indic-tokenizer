#!/usr/bin/env python3

import sys
import os.path
from logger import get_logger

if len(sys.argv) < 4:
    print("requires <folder|file> <vsize> <outbase>")
    sys.exit(0)

_path = sys.argv[1]
_vsize =int(sys.argv[2])
_outbase = sys.argv[3]

logger = get_logger("indic-bert-tokenizer-builder")

# check if the file exists
if not os.path.exists(_path):
    logger.error(f"{_path=} does not exist!")
    sys.exit(0)

import os
from glob import glob
from indic_unicode_mapper import IndicUnicodeMapper

# collect the input data file paths.
# we use only the *.txt files if a folder is presented.
files = []
if os.path.isdir(_path):
    files = [y for x in os.walk(_path) for y in glob(os.path.join(x[0], '*.txt'))]
elif os.path.isfile(_path):
    files.append(_path)
else:
    logger.error(f"{_path=} is not a valid file or directory.")
    sys.exit(0)

# sanity check.
if len(files) <= 0:
    logger.error("no valid input data files to process.")
    sys.exit(0)

from indic_bert_tokenizer import IndicBertWordPieceTokenizer
tok = IndicBertWordPieceTokenizer.build_model(files, vocab_size=_vsize, model_dir=_outbase, human_readable=True)

toks = tok.encode("தமிழ் மொழி ஒரு இனிய மொழி. മലയാളം ഭാഷയും இனிய ഭാഷ.")
print(toks.ids)
print(toks.tokens)
print([tok.decode_string(tok) for tok in toks.tokens])
