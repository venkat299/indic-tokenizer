import argparse
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)
from indic_bert_tokenizer import IndicBertWordPieceTokenizer
from indic_unicode_mapper import IndicUnicodeMapper


def main():
    parser = argparse.ArgumentParser(description="Pre-train a tiny GPT model on Tamil text")
    parser.add_argument("--samples", type=int, default=1000, help="number of training samples")
    parser.add_argument("--output_dir", type=str, default="tamil-pretrained", help="where to save the model")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="path to Indic tokenizer vocab file",
    )
    args = parser.parse_args()

    dataset = load_dataset("ai4bharat/IndicCorp", "ta", split=f"train[:{args.samples}]")

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

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    config = GPT2Config(
        vocab_size=hf_tokenizer.vocab_size,
        n_positions=256,
        n_ctx=256,
        n_embd=128,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=1000,
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
