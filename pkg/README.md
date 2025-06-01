# indic-tokenizer
Python modules for tokenizing Indian languages (only Tamil is implemented for now.)

## Features

- Tokenizes text in Indian languages (currently supports only Tamil).
- Simple API for integrating tokenization into your Python projects.

## Installation

```bash
pip install -i https://test.pypi.org/simple/ indic-tokenizer

```

## Build the model

Assuming a folder or a file containing the `text` content for tokenization, the first step is to map the graphemes into singular unicodes.

```python
# Indic Unicode Mapper maps the sequence of unicode that constitute a grapheme 
# into a singular unicode in the 0xE00X range.
from indic_tokenizer import IndicUnicodeMapper
mapper = IndicUnicodeMapper()
# encode the graphemes.
# LANG = "ta"
encoded_text = mapper.encode(text=text, lang=LANG)
```

We then build a BERT WPE tokenizer on the encoded text.

```python
# instantiate the BERT WPE tokenizer module.
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

# setup the Vocabulary size requirement
VOCAB_SIZE = 3000

# train the algorithm
tokenizer.train(files=ENCODED-FILES-FOLDER, vocab_size=VOCAB_SIZE, min_frequency=2,
                limit_alphabet=512, wordpieces_prefix='##',
                special_tokens=spl_tokens)

# save the tokenization model.
# this line should create a file with the name f"{LANG}-vocab.txt"
tokenizer.save_model('.', LANG)
TOKENIZER_MODEL = f"{LANG}-vocab.txt"               
```

## Usage

```python
from indic_tokenizer import IndicBertWordPieceTokenizer

tokenizer = IndicBertWordPieceTokenizer(TOKENIZER_MODEL)
text = "வணக்கம்! இது ஒரு எடுத்துக்காட்டு."
tokens = tokenizer.encode(text)
print(tokens.ids)
>>> [3, 73, 441, 1067, 5, 815, 692, 2747, 420, 682, 9, 1]
print(tokens.tokens)
>>> ['[cls]', 'வ', '##ண', '##\ue011க\ue10e', '!', 'இ\ue0a5', 'ஒ\ue12f', 'எ\ue077\ue0b2\ue0a5\ue011', '##\ue001', '##\ue084\ue077', '.', '[sep]']
print([tokenizer.decode_string(tok) for tok in toks.tokens])
>>> ['[cls]', 'வ', '##ண', '##க்கம்', '!', 'இது', 'ஒரு', 'எடுத்துக்', '##கா', '##ட்டு', '.', '[sep]']
# get the decoded string from the tokenizer ids.
tokenizer.decode(toks.ids)
>>> 'வணக்கம்! இது ஒரு எடுத்துக்காட்டு.'
```

## Tutorial

A jupyter notebook [tutorial](https://github.com/sudarsun/indic-tokenizer/blob/main/tutorial.ipynb) is also available to build a tokenizer model followed by loading and using for tokenization of Tamil texts.

## Supported Languages

- Tamil (more languages planned for future releases)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes or new features.

## License

This project is licensed under the MIT License.
