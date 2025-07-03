# indic-tokenizer
Python modules for tokenizing Indian languages (only Tamil & Malayalam are implemented for now.)

## Features

- Tokenizes text in Indian languages (currently supports only Tamil & Malayalam).
- Simple API for integrating tokenization into your Python projects.

## Installation

```bash
pip install -r requirements.txt

```

## Build the model

Assuming a folder or a file containing the `text` content for tokenization (say `ENCODED-FILES-FOLDER`), the first step is to map the graphemes into singular unicodes.

```python
# Set up the Vocabulary size requirement
VOCAB_SIZE = 3000
# set up the output base directory
# Typically, the model gets saved as OUTBASE_DIR/indic-bert-tokenizer-vocab.txt
# Additionally, if we also need the human-readable vocabulary, the file gets saved as OUTBASE_DIR/indic-bert-tokenizer-vocab.indic.txt
OUTBASE_DIR = "."

# Indic tokenizer uses Indic Unicode mapper that maps the sequence of Unicode that constitutes a grapheme 
# into a singular Unicode in the 0xE00X range.
from indic_bert_tokenizer import IndicBertWordPieceTokenizer
tok = IndicBertWordPieceTokenizer.build_model(files, vocab_size=VOCAB_SIZE, model_dir=OUTBASE_DIR, human_readable=True)      
```

#### `TOKENIZE_MODEL` is typically `OUTBASE_DIR/indic-bert-tokenizer-vocab.txt`

## Usage

```python
from indic_bert_tokenizer import IndicBertWordPieceTokenizer

TOKENIZER_MODEL = OUTBASE_DIR + "/indic-bert-tokenizer-vocab.txt"
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

A Jupyter notebook tutorial [tutorial](/tutorial.ipynb) is also available to build a tokenizer model, followed by loading and using it for tokenizing Tamil and Malayalam texts.


## Supported Languages

- Tamil
- Malayalam
- (more languages planned for future releases)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes or new features.

## License

This project is licensed under the MIT License.
