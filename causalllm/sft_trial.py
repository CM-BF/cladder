import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

model_name = "NousResearch/Llama-2-7b-chat-hf"  # The model that you want to train from the Hugging Face hub
# model_name = "NousResearch/Llama-2-7b-chat-hf"  # The model that you want to train from the Hugging Face hub
dataset_name = "BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1"  # The instruction dataset to use
new_model = "llama-persian-catalog-generator"  # The name for fine-tuned LoRA Adaptor

# LoRA parameters
lora_r = 64
lora_alpha = lora_r * 2
lora_dropout = 0.1
target_modules = ["q_proj", "v_proj", "k_proj"]

# QLoRA parameters
load_in_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = False

# TrainingArguments parameters
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
learning_rate = 0.00015
weight_decay = 0.01
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

# Dataset parameters
use_special_template = True
response_template = " ### Answer:"
instruction_prompt_template = '"### Human:"'
use_llama_like_model = True

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")
percent_of_train_dataset = 0.95
other_columns = [i for i in dataset.column_names if i not in ["instruction", "output"]]
dataset = dataset.remove_columns(other_columns)
split_dataset = dataset.train_test_split(
    train_size=int(dataset.num_rows * percent_of_train_dataset), seed=19, shuffle=False
)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")

# Load LoRA configuration
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# Load QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device_map)
model.config.use_cache = False

# Set training parameters
training_arguments = SFTConfig(
    output_dir=new_model,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    gradient_checkpointing=gradient_checkpointing,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    max_seq_length=max_seq_length,
    packing=packing,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training: Qwen's default is right
if not tokenizer.chat_template:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

def special_formatting_prompts(example):
    # output_texts = []
    # for i in range(len(example["instruction"])):
    #     text = f"{instruction_prompt_template}{example['instruction'][i]}\n{response_template} {example['output'][i]}"
    #     output_texts.append(text)
    # return output_texts
    return f"{instruction_prompt_template}{example['instruction']}\n{response_template} {example['output']}"


def normal_formatting_prompts(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        chat_temp = [
            {"role": "system", "content": example["instruction"][i]},
            {"role": "assistant", "content": example["output"][i]},
        ]
        text = tokenizer.apply_chat_template(chat_temp, tokenize=False)
        output_texts.append(text)
    return output_texts

if use_special_template:
    formatting_func = special_formatting_prompts
    if use_llama_like_model:
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
else:
    formatting_func = normal_formatting_prompts

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    data_collator=collator,
    processing_class=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save fine tuned Lora Adaptor
trainer.model.save_pretrained(new_model)