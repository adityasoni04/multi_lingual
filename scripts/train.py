# scripts/train.py (Massively Multilingual - Optimized for High-End GPU)

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import langcodes

def get_lang_name(code):
    """Converts a language code like 'hin_Deva' to its name 'Hindi'."""
    try:
        return langcodes.Language.get(code.split('_')[0]).display_name()
    except (AttributeError, langcodes.LanguageTagError):
        return code

def main():
    # --- 1. Configuration ---
    MODEL_CHECKPOINT = "google/flan-t5-xl"
    MODEL_SAVE_PATH = f"./models/{MODEL_CHECKPOINT.replace('/', '-')}-multilingual"

    # --- Hardware Check: Allow Ampere (8.x) and newer (e.g., Blackwell 9.x, Hopper 9.x) ---
    if not torch.cuda.is_available():
        raise SystemError("CUDA device not found. Please ensure an NVIDIA GPU is installed and CUDA drivers are correctly configured.")
    major = torch.cuda.get_device_properties(0).major
    if major < 8:
        raise SystemError("This script requires an NVIDIA GPU with Ampere (compute capability 8.0) or newer (Hopper/Blackwell, compute capability 9.0+). Detected: "
                          f"{torch.cuda.get_device_name(0)}, Compute Capability: {major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"✅ Powerful GPU found! Using device: {torch.cuda.get_device_name(0)}")

    # --- 2. Load Model & Tokenizer ---
    print(f"🔄 Loading powerful model '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("✅ Model loaded.")

    # --- 3. Load, Process, and Combine Datasets ---
    print("🔄 Preparing multilingual datasets...")

    # --- Translation Dataset (NLLB) ---
    target_langs = [
        "asm_Beng", "ben_Beng", "doi_Deva", "eng_Latn", "gom_Deva", "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "kas_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "mni_Beng", "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck", "snd_Arab", "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab",
        "deu_Latn", "fra_Latn"
    ]
    translation_ds = load_dataset("allenai/nllb-seed", split="train").filter(
        lambda x: x['translation']['src_lang'] in target_langs and x['translation']['tgt_lang'] in target_langs
    )

    def preprocess_translation(example):
        src_lang_name = get_lang_name(example['translation']['src_lang'])
        tgt_lang_name = get_lang_name(example['translation']['tgt_lang'])
        prefix = f"Translate {src_lang_name} to {tgt_lang_name}: "
        inputs = [prefix + example['translation']['src_text']]
        targets = [example['translation']['tgt_text']]
        return tokenizer(inputs, text_target=targets, max_length=256, truncation=True)

    translation_tokenized = translation_ds.map(preprocess_translation, remove_columns=translation_ds.column_names)

    # --- Transliteration Dataset (IndicXlit) ---
    all_translit_datasets = []
    lang_pairs = ["as-en", "bn-en", "gu-en", "hi-en", "kn-en", "ml-en", "mr-en", "or-en", "pa-en", "ta-en", "te-en", "ur-en"]
    for pair in lang_pairs:
        lang_code, _ = pair.split('-')
        lang_name = get_lang_name(lang_code)

        for direction in ["native_to_eng", "eng_to_native"]:
            ds = load_dataset("ai4bharat/IndicXlit", pair, split=direction)
            src_lang, tgt_lang = (lang_name, "English") if direction == "native_to_eng" else ("English", lang_name)

            def preprocess_transliteration(example, src=src_lang, tgt=tgt_lang):
                prefix = f"Transliterate {src} to {tgt}: "
                inputs = [prefix + example['source']]
                targets = [example['target']]
                return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)

            tokenized_ds = ds.map(preprocess_transliteration, remove_columns=ds.column_names)
            all_translit_datasets.append(tokenized_ds)

    # --- Combine and Shuffle ---
    combined_dataset = concatenate_datasets([translation_tokenized] + all_translit_datasets).shuffle(seed=42)
    print(f"✅ All datasets combined and processed. Total examples: {len(combined_dataset)}")

    # --- 4. Training ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        num_train_epochs=1,
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=50,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    print("🚀 Starting large-scale training on your powerful GPU...")
    trainer.train()
    print("🎉 Training finished!")

    # --- 5. Save Final Model ---
    print(f"💾 Saving final model to {MODEL_SAVE_PATH}...")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

if __name__ == "__main__":
    main()
