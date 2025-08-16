import argparse
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from indic_bert_tokenizer import IndicBertWordPieceTokenizer
from indic_unicode_mapper import IndicUnicodeMapper


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the Tamil model on song lyrics")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset containing Tamil lyrics")
    parser.add_argument("--model_dir", type=str, default="tamil-pretrained", help="path to the pretrained model")
    parser.add_argument("--output_dir", type=str, default="tamil-lyrics-model", help="where to save the fine-tuned model")
    parser.add_argument("--samples", type=int, default=1000, help="number of lyric samples to use")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="path to Indic tokenizer vocab file",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=f"train[:{args.samples}]")
    indic_tok = IndicBertWordPieceTokenizer(args.tokenizer_path)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=indic_tok._tokenizer._tokenizer,
        bos_token="[cls]",
        eos_token="[sep]",
        unk_token="[unk]",
        pad_token="[pad]",
        mask_token="[mask]",
    )
    mapper = IndicUnicodeMapper()

    def tokenize_function(examples):
        mapped = [mapper.encode(t) for t in examples["text"]]
        return hf_tokenizer(mapped)

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=500,
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    hf_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
