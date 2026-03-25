import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth not installed. Will fallback to standard transformers if used (not recommended for Grubbot).")

def load_model(model_name: str, max_seq_length: int = 2048):
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is required for Grubbot finetuning. Please install it.")
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def formatting_prompts_func(tokenizer):
    def wrapper(example):
        texts = []
        for i in range(len(example['messages'])):
            # Extremely simplified text generation - assumes chat template applies 
            # In a real scenario, applying tokenizer.apply_chat_template is preferred
            msgs = example['messages'][i]
            expected_call = example['expected_tool_call'][i]
            
            # Format expected call as the assistant's output
            assistant_response = f'{{"name": "{expected_call["name"]}", "arguments": {expected_call["arguments"]}}}'
            
            conversation = [
                {"role": "user", "content": msgs[0]["content"]},
                {"role": "assistant", "content": assistant_response}
            ]
            
            text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
        return {"text": texts}
    return wrapper

def prepare_dataset(train_path: str, tokenizer):
    dataset = load_dataset("json", data_files=train_path, split="train")
    formatter = formatting_prompts_func(tokenizer)
    dataset = dataset.map(formatter, batched=True)
    return dataset

def train(model, tokenizer, dataset, output_dir: str):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
        ),
    )
    trainer.train()
    return trainer

def save_checkpoint(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
