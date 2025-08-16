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

## Tamil Language Model and Lyrics Generator

This project now includes utilities for building a tiny Tamil language model and generating song lyrics.

### 1. Pre-train a Tiny Tamil Model

Download a small portion of the open [IndicCorp](https://huggingface.co/datasets/ai4bharat/IndicCorp) dataset and run:

```bash
python scripts/pretrain_tamil_gpt.py \
  --samples 1000 \
  --output_dir tamil-pretrained \
  --tokenizer_path path/to/indic-bert-tokenizer-vocab.txt
```

The script trains a compact GPT-2 style model from scratch using the provided Indic tokenizer vocabulary and saves it to `tamil-pretrained/`.

### 2. Fine-tune on Tamil Song Lyrics

Obtain a Tamil lyrics dataset from [Hugging Face Datasets](https://huggingface.co/datasets) and fine-tune the pretrained model:

```bash
python scripts/fine_tune_lyrics.py \
  --dataset_name <dataset> \
  --model_dir tamil-pretrained \
  --output_dir tamil-lyrics-model \
  --samples 1000 \
  --tokenizer_path path/to/indic-bert-tokenizer-vocab.txt
```

Replace `<dataset>` with the dataset identifier (for example, `your-username/tamil-lyrics`). The fine-tuned model and tokenizer are saved to `tamil-lyrics-model/`.

### 3. Web Interface for Lyrics Generation

A simple [Gradio](https://gradio.app) application generates song lyrics for a given theme:

```bash
python app.py
```

Enter a theme and the app will produce approximately three minutes of Tamil lyrics using the fine-tuned model.


## Supported Languages

- Tamil
- Malayalam
- (more languages planned for future releases)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes or new features.

## License

This project is licensed under the MIT License.
