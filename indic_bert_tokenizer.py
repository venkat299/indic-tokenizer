#/usr/bin/env python3

from tokenizers import BertWordPieceTokenizer
from indic_unicode_mapper import IndicUnicodeMapper

# Extension of the Bert WP Tokenizer in the Indic context (Tamil for starters)
class IndicBertWordPieceTokenizer:
    __unk_token = "[unk]"  # token for unknown words
    __cls_token = "[cls]"
    __sep_token = "[sep]"
    __mask_token = "[mask]"
    __pad_token = "[pad]"
    
    def __init__(self, model_path:str):
        # initialize our indic unicode mapper
        self._mapper = IndicUnicodeMapper()
        # create our base BERT tokenizer
        self._tokenizer = BertWordPieceTokenizer.from_file(model_path,
                                                           clean_text=False, 
                                                           handle_chinese_chars=True,
                                                           strip_accents=False,
                                                           lowercase=False,
                                                           sep_token = self.__sep_token, 
                                                           unk_token= self.__unk_token, 
                                                           mask_token=self.__mask_token, 
                                                           cls_token=self.__cls_token, 
                                                           pad_token=self.__pad_token)

    # method to encode the indic text
    def encode(self, text:str, lang="ta"):
        # map the indic text to higher unicodes
        norm_text = self._mapper.encode(text=text, lang=lang)
        # use the base tokenizer to tokenize the mapped text
        return self._tokenizer.encode(norm_text)

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
