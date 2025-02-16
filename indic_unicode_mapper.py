import pygtrie

class IndicUnicodeMapper:
    __tamil_vowels = ['\u0BBE', '\u0BBF', '\u0BC0', '\u0BC1', '\u0BC2', '\u0BC6', '\u0BC7', '\u0BC8', '\u0BCA', ['\u0BBE', '\u0BC6'], ['\u0BC6', '\u0BBE'], '\u0BCB', '\u0BD7', '\u0BCC', ['\u0BC6', '\u0BD7'], ['\u0BD7', '\u0BC6'], '\u0BCD', ['\u0BBE', '\u0BC7'], ['\u0BC7', '\u0BBE'],  ['\u0BC1', '\u0BBE'], ['\u0BC1', '\u0BC1'],['\u0BCB', '\u0BBF'],['\u0BCA', '\u0BBF']] 
    __tamil_consonants = ['\u0B95', '\u0B99', '\u0B9A', '\u0B9C', '\u0B9E', '\u0B9F', '\u0BA3', '\u0BA4', '\u0BA8', '\u0BA9', '\u0BAA', '\u0BAE', '\u0BAF', '\u0BB0', '\u0BB1', '\u0BB2', '\u0BB3', '\u0BB4', '\u0BB5', '\u0BB6', '\u0BB7', '\u0BB8', '\u0BB9']
    # some usage conventions need fixing.
    __tamil_replacements = {"ாி":"ரி", "ா்":"ர்", "ௗ்":"ள்"}

    # order of loading the languages.
    __indic_languages = ['ta']  
    # put all the indian vowel consonant pairs here
    __indic_symbols = {"ta":(__tamil_vowels, __tamil_consonants)}
    # the starting point of the mapped symbols
    __start_unicode = 0xE001
    # max grapheme length for a language
    __max_length = {"ta":3}

    # cache of language specific vowels.
    __all_vowels = {}

    def __init__(self):
        self.__forward = pygtrie.CharTrie()
        self.__reverse = {}

        _index = self.__start_unicode
        for lang in self.__indic_languages:
            (v, c) = self.__indic_symbols[lang]
            for _c in c:
                for _v in v:
                    _s = _c
                    if type(_v) == list:
                        _s += "".join(_v)
                    else:
                        _s += _v
                    # convert the unicode index into an unicode character
                    mapped_unicode = chr(_index)
                    self.__forward[_s] = mapped_unicode
                    self.__reverse[mapped_unicode] = _s
                    _index += 1
            # create a cache of vowels
            cache = set()
            for v_ in v:
                if type(v_) == list:
                    for v__ in v_:
                        cache.add(v__)
                else:
                    cache.add(v_)
            # populate the language specific vowels
            self.__all_vowels[lang] = cache

    # create the normalization and denormalization tsv file for sentencepiece tokenizer
    def generate_norm_rule_tsv(self, path:str):
        # open the rules files.
        with open(path + "-norm.tsv", "w") as fn:
            with open(path + "-denorm.tsv", "w") as fd:
                # iterate over the mapped unicode keys
                for key in self.__reverse.keys():
                    # create the key string in unicode hex
                    key_string = hex(ord(key))
                    # create the value string in unicode hex
                    val_string = " ".join(map(lambda x: hex(ord(x)), self.__reverse[key]))

                    # add the records to the rules files.
                    fd.write(key_string + '\t' + val_string + '\n')
                    fn.write(val_string + '\t' + key_string + '\n')
                # close the file
                fd.close()
            # close the file
            fn.close()
        return True

    # check if the given text contains language vowels.
    def is_consistent(self, text:str, lang="ta"):
        if lang not in self.__all_vowels:
            raise ValueError(f"unknown language {lang=}")
        
        # scan through the text for left out vowels.
        # stop on the first find.
        for symbol in text:
            if symbol in self.__all_vowels[lang]:
                return False
        # all good here.
        return True
    
    # replace broken strings into correct formats
    def __normalize(self, text:str) -> str:
        for item in self.__tamil_replacements.keys():
            text = text.replace(item, self.__tamil_replacements[item])
        return text
 
    # encode the supplied ucs-2 indic string into unicode mapped ucs-2 string
    def encode(self, text:str, lang="ta"):
        if lang not in self.__max_length:
            raise ValueError(f"unknown language {lang=}")
        
        # normalize the text to get rid of inconsistencies
        text = self.__normalize(text)

        # get the length of the length
        length = len(text)
        # get the language specific chunk length
        max_chunk_length = self.__max_length[lang]

        # encode is a linear complexity algorithm
        # the hope is that the calling layer will use parallel processing.
        current = 0
        output:str = ""
        while current < length:
            success = False
            for chunk_length in range(max_chunk_length, 1, -1):
                substr = text[current:current+chunk_length]
                if substr in self.__forward:
                    mapped_str = self.__forward[substr]
                    current += chunk_length
                    success = True
                    output += mapped_str
                    break
            if not success:
                if text[current] != '\u0bcd':
                    output += text[current]
                current += 1
        
        return output

    # decode the mapped text to the original form.
    def decode(self, text:str):
        # decode is a linear complexity algorithm
        # the hope is that the calling layer will use parallel processing.
        return "".join(map(lambda x: self.__reverse[x] if x in self.__reverse else x, text))
