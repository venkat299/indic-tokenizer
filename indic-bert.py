#!/usr/bin/env python3

import sys
import os.path

if len(sys.argv) < 5:
    print("requires <folder|file> <lang=ta|en> <vsize> <outbase>")
    sys.exit(0)

_path = sys.argv[1]
_lang = sys.argv[2]
_vsize =int(sys.argv[3])
_outbase = sys.argv[4]

# check if the file exists
if not os.path.exists(_path):
    print(f"{_path=} does not exist!")
    sys.exit(0)

import os
from glob import glob
import tempfile
from indic_unicode_mapper import IndicUnicodeMapper

# collect the input data file paths.
# we use only the *.txt files if a folder is presented.
files = []
if os.path.isdir(_path):
    files = [y for x in os.walk(_path) for y in glob(os.path.join(x[0], '*.txt'))]
elif os.path.isfile(_path):
    files.append(_path)
else:
    print(f"{_path=} is not a usable path.")
    sys.exit(0)

# sanity check.
if len(files) <= 0:
    print("no valid input data files to process.")
    sys.exit(0)

import multiprocessing

tmpdir = f"/tmp/mapped-{_lang}"
os.makedirs(tmpdir, exist_ok=True)

er = open('/tmp/errored.txt', "w")

if _lang == "ta":
    mapper = IndicUnicodeMapper()
    nfiles = []

    for file in files:
        fname = os.path.basename(file)
        fpath = tmpdir + "/" + fname
        print(f"mapping {file} -> {fpath}")
        with open(fpath, "w") as fw:
            with open(file, "r") as fh:
                lines = fh.readlines()
                fh.close()

                tpool = multiprocessing.Pool()
                elines = tpool.map(mapper.encode, lines)
                consistency_check = tpool.map(mapper.is_consistent, elines)
                tpool.close()

                for i in range(len(consistency_check)):
                    if not consistency_check[i]:
                        er.write(lines[i] + "\n")
                        er.write(elines[i] + "\n")

                etext = "\n".join(elines)
            fw.write(etext)
            fw.close()
            nfiles.append(fpath)

    files = nfiles

from tokenizers import BertWordPieceTokenizer

cls_token = "[cls]"
sep_token = "[sep]"
mask_token = "[mask]"
pad_token = "[pad]"
unk_token = "[unk]"
spl_tokens = ["[unk]", "[sep]", "[mask]", "[cls]", "[pad]"]  # special tokens
tokenizer = BertWordPieceTokenizer(clean_text=False, 
                                   handle_chinese_chars=True, 
                                   strip_accents=False,
                                   lowercase=False,
                                   sep_token=sep_token, unk_token=unk_token, 
                                   mask_token=mask_token, cls_token=cls_token, pad_token=pad_token)

tokenizer.train(files=files, vocab_size=_vsize, min_frequency=2,
                limit_alphabet=512, wordpieces_prefix='##',
                special_tokens=spl_tokens)

tokenizer.save_model('.', _outbase)

