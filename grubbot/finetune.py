import json
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel

def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # CPU fallback: only use device_map="auto" when CUDA is available
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer

def load_from_checkpoint(base_model_name: str, checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=True)
    
    return model, tokenizer

def formatting_prompts_func(tokenizer):
    def wrapper(example):
        texts = []
        for i in range(len(example['messages'])):
            msgs = example['messages'][i]
            expected_call = example['expected_tool_call'][i]
            
            # Safely extract tools_schema without IndexError
            tools_schema = []
            if 'tools' in example and example['tools']:
                tools_schema = example['tools'][i]
            
            # 1. Add system prompt with tool schema so the model knows what tools exist
            tools_json = json.dumps(tools_schema, indent=2)
            system_prompt = f"You are a helpful assistant with access to the following tools:\n{tools_json}\n\nWhen the user's request matches a tool, respond with a JSON object containing 'name' and 'arguments'. If no tool matches, respond with 'null'."
            
            # 2. Fix the malformed JSON bug (using json.dumps instead of f-string dict repr)
            if expected_call is None:
                assistant_response = "null"
            else:
                assistant_response = json.dumps({"name": expected_call["name"], "arguments": expected_call["arguments"]})
            
            conversation = [
                {"role": "system", "content": system_prompt},
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

def train(model, tokenizer, dataset, output_dir: str, iteration: int = 1):
    base_lr = 2e-4
    current_lr = base_lr / (iteration ** 0.5)

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_length=512,  # Reduced for CPU efficiency
        packing=False,
        per_device_train_batch_size=1,  # CPU safety
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=current_lr,
        fp16=False,  # Forced off for CPU
        bf16=False,  # Forced off for CPU
        logging_steps=1,
        optim="adamw_torch",  # Standard PyTorch optimizer (CPU safe)
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        use_cpu=not torch.cuda.is_available(),  # Force CPU when no GPU
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )
    trainer.train()
    return trainer

def save_checkpoint(model, tokenizer, path: str):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)