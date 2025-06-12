#/usr/bin/env python3

# @author: Sudarsun S
# @date: 2025-06-12
# Implementation of an Indic BERT WordPiece Tokenizer using the Indic Unicode Mapper.
# -*- coding: utf-8 -*-
"""
indic_bert_tokenizer.py
This module implements an Indic BERT WordPiece Tokenizer using the Indic Unicode Mapper. It provides methods to build the tokenizer model from text files and to encode/decode Indic text.
"""

from tokenizers.implementations import BertWordPieceTokenizer
from indic_unicode_mapper import IndicUnicodeMapper
import tempfile
import os
import multiprocessing
import shutil
from logger import get_logger

# Extension of the Bert WP Tokenizer in the Indic context (Tamil for starters)
class IndicBertWordPieceTokenizer:
    __unk_token = "[unk]"  # token for unknown words
    __cls_token = "[cls]"
    __sep_token = "[sep]"
    __mask_token = "[mask]"
    __pad_token = "[pad]"

    @staticmethod
    def build_model(files:list[str], model_dir:str="./", vocab_size:int=30000, min_frequency:int=2, human_readable:bool=False):
        """
        Build the vocabulary for the tokenizer from the given files.
        :param files: List of files to build the vocabulary from.
        :param vocab_size: Size of the vocabulary to build.
        :param min_frequency: Minimum frequency of words to include in the vocabulary.
        """
        logger = get_logger("IndicBERTWPETokenizer.build_model")

        # create the mapper object
        mapper = IndicUnicodeMapper()
        # create a temporary directory to store the mapped files
        tmpdir = tempfile.TemporaryDirectory().name
        os.makedirs(tmpdir, exist_ok=True)
        logger.info(f"Using temporary directory {tmpdir} for mapped files.")
        # list of mapped files.
        nfiles = []
        logger.info(f"Processing {len(files)} files for vocabulary building.")       
        for file in files:
            fname = os.path.basename(file)
            fpath = tmpdir + "/" + fname
            logger.info(f"Processing file {file} -> {fpath}")
            # open the file and map the contents to higher unicode values
            # this is to ensure that the tokenizer can handle the Indic text properly.
            with open(fpath, "w") as fw:
                # read the file line by line and encode each line
                with open(file, "r") as fh:
                    lines = fh.readlines()
                    fh.close()
                    # create a pool of workers to encode the lines
                    tpool = multiprocessing.Pool()
                    elines = tpool.map(mapper.encode, lines)
                    tpool.close()
                    # combined the encoded lines into a single string
                    etext = "\n".join(elines)
                # write the encoded text to the file
                fw.write(etext)
                fw.close()
            # add the file path to the list of mapped files
            nfiles.append(fpath)
        
        # create the tokenizer instance
        # we use the BertWordPieceTokenizer from the tokenizers library
        tokenizer = BertWordPieceTokenizer(clean_text=False, handle_chinese_chars=True, 
                                           strip_accents=False, lowercase=False,
                                           sep_token=IndicBertWordPieceTokenizer.__sep_token, unk_token=IndicBertWordPieceTokenizer.__unk_token, 
                                           mask_token=IndicBertWordPieceTokenizer.__mask_token, cls_token=IndicBertWordPieceTokenizer.__cls_token, 
                                           pad_token=IndicBertWordPieceTokenizer.__pad_token)
        
        # train the tokenizer on the provided files
        logger.info(f"Training tokenizer on {len(nfiles)} files with vocab size {vocab_size} and min frequency {min_frequency}")
        tokenizer.train(files=nfiles, vocab_size=vocab_size, min_frequency=min_frequency,
                        limit_alphabet=512, wordpieces_prefix='##',
                        special_tokens=[IndicBertWordPieceTokenizer.__unk_token, IndicBertWordPieceTokenizer.__sep_token, 
                                        IndicBertWordPieceTokenizer.__mask_token, IndicBertWordPieceTokenizer.__cls_token, 
                                        IndicBertWordPieceTokenizer.__pad_token])
        # save the tokenizer model
        _outbase = "indic-bert-tokenizer"
        logger.info(f"Saving tokenizer model to {model_dir}/{_outbase}-vocab.txt")
        tokenizer.save_model(model_dir, _outbase)
        # clean up the temporary directory
        shutil.rmtree(tmpdir, ignore_errors=True)

        # let's create another vocabulary for humans to understand.
        if human_readable:
            logger.info(f"Creating human readable vocabulary file at {model_dir}/{_outbase}-vocab.indic.txt")
            # open the vocabulary file and map the tokens to indic unicode
            # this is to ensure that the vocabulary file is readable by humans
            with open(model_dir + "/" + _outbase + "-vocab.txt", "r") as fin:
                items = fin.readlines()
                fin.close()
                with open(model_dir + "/" + _outbase + "-vocab.indic.txt", "w") as fout:
                    mapped = "".join(map(mapper.decode, items))
                    fout.write(mapped)
                    fout.close()

        # return the tokenizer instance with the vocabulary file
        return IndicBertWordPieceTokenizer(model_path=os.path.join(model_dir, _outbase + "-vocab.txt"))
                    
    def __init__(self, model_path:str):
        # initialize our indic unicode mapper
        self._mapper = IndicUnicodeMapper()
        # create our base BERT tokenizer
        self._tokenizer = BertWordPieceTokenizer.from_file(model_path, clean_text=False, handle_chinese_chars=True,
                                                           strip_accents=False, lowercase=False,
                                                           sep_token = self.__sep_token, unk_token= self.__unk_token, 
                                                           mask_token=self.__mask_token, cls_token=self.__cls_token, pad_token=self.__pad_token)

    # method to encode the indic text
    def encode(self, text:str, lang="ta"):
        # map the indic text to higher unicodes
        norm_text = self._mapper.encode(text=text, lang=lang)
        # use the base tokenizer to tokenize the mapped text
        return self._tokenizer.encode(norm_text)
    
    def tokenize(self, text:str, lang="ta"):
        """
        Tokenize the given text using the tokenizer.
        :param text: Text to tokenize.
        :param lang: Language of the text (default is Tamil).
        :return: List of tokens.
        """
        # encode the text to get the token ids
        # return the tokens from the encoded object
        return self.encode(text, lang)

    # method to decode the indic text
    def decode(self, ids:list[int]):
        # decode the token ids into mapped tokens
        decoded = self._tokenizer.decode(ids)
        # denormalize the mapped tokens to indic language
        norm_text = self._mapper.decode(decoded)
        return norm_text
    
    # method to decode the unicode mapping for strings.
    def decode_string(self, text:str):
        return self._mapper.decode(text)
