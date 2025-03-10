import operator
from pathlib import Path
from typing import Literal, Any, List, TypedDict, Annotated, Dict, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from itertools import product
from datasets import Dataset
from tqdm import tqdm

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

from causalllm.data import DataFileList
from causalllm.evaluate import Scorer
from causalllm.prompt_utils import partial_replace, TextInterfaceForLLMs
from causalllm.definitions import ROOT_PATH, missing_step, paraphrase_i


class CausalTrainer:
    """Class for training causal inference models using Hydra for configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Setting up paths
        self.output_dir = Path.cwd()  # Hydra changes working directory
        print(f"Working directory: {self.output_dir}")

        # Save the config for reproducibility
        with open(self.output_dir / "config_dump.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    def train_sft(self):
        """Train a model using Supervised Fine-Tuning with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        new_model = self.cfg.model.new_model
        reasoning = self.cfg.experiment.reasoning
        ask_about = self.cfg.experiment.ask_about
        enable_cot = self.cfg.experiment.enable_cot
        enable_fewshot = self.cfg.experiment.enable_fewshot
        given_cot_until_step = self.cfg.experiment.given_cot_until_step

        # Setup output files
        write_out_file = self.output_dir / 'output.csv'

        # Create text interface for prompt composition
        text_interface = TextInterfaceForLLMs(
            write_out_file,
            ask_about=ask_about,
            enable_fewshot=enable_fewshot,
            enable_cot=enable_cot,
            given_cot_until_step=given_cot_until_step
        )

        # Load dataset
        df_list = DataFileList(ask_about=ask_about)

        # Process each dataset file
        for data_file_obj in df_list.data_objs:
            self._perform_sft(data_file_obj, model_name, reasoning, text_interface, new_model)

        # Optional: evaluate the model
        if self.cfg.output.evaluate:
            scorer = Scorer([str(write_out_file)], ask_about)

    def _perform_sft(self, data_file_obj, model_name, reasoning, text_interface, new_model):
        """Perform the actual SFT training for a given dataset"""
        # Extract configurations
        lora_config = self.cfg.lora
        quant_config = self.cfg.quantization
        training_config = self.cfg.training
        dataset_config = self.cfg.dataset

        # Setup device map
        device_map = {"": 0} if training_config.device_map == 0 else training_config.device_map

        # Prepare the dataset
        text_interface.prepare_prompt_sft(data_file_obj.data, reasoning=reasoning)
        data = text_interface.data_in
        all_samples = list(map(lambda x: {k: v for k, v in x.items() if k != 'old'}, data))
        dataset = Dataset.from_list(all_samples)

        percent_of_train_dataset = dataset_config.percent_of_train_dataset
        # "prompt" is a special column key for SFTTrainer's data preprocessing
        dataset = dataset.rename_columns({"prompt": "instruction", "response": "output"})
        other_columns = [i for i in dataset.column_names if i not in ["instruction", "output"]]
        dataset = dataset.remove_columns(other_columns)

        # Format dataset for model input
        if not dataset_config.use_special_template:
            dataset = dataset.map(self._format_transform, desc="Formatting dataset to ChatML")

        # Split dataset
        split_dataset = dataset.train_test_split(
            train_size=int(dataset.num_rows * percent_of_train_dataset),
            seed=dataset_config.seed,
            shuffle=dataset_config.shuffle
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")

        # Setup LoRA configuration
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            target_modules=list(lora_config.target_modules),
        )

        # Setup quantization configuration
        compute_dtype = getattr(torch, quant_config.compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.load_in_4bit,
            bnb_4bit_quant_type=quant_config.quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.use_double_quant,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=self.cfg.model.trust_remote_code
        )
        model.config.use_cache = False

        # Setup training arguments
        training_arguments = SFTConfig(
            output_dir=new_model,
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            optim=training_config.optim,
            save_steps=training_config.save_steps,
            logging_steps=training_config.logging_steps,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            max_steps=training_config.max_steps,
            warmup_ratio=training_config.warmup_ratio,
            gradient_checkpointing=training_config.gradient_checkpointing,
            group_by_length=training_config.group_by_length,
            lr_scheduler_type=training_config.lr_scheduler_type,
            max_seq_length=training_config.max_seq_length,
            packing=training_config.packing,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.cfg.model.trust_remote_code
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        # Setup data collator and formatting function
        formatting_func = None
        collator = None

        if dataset_config.use_special_template:
            formatting_func = self._special_formatting_prompts
            response_template_tokens = tokenizer.encode(
                dataset_config.response_template,
                add_special_tokens=False
            )
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_tokens,
                tokenizer=tokenizer
            )

        # Create trainer
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

        # Save fine-tuned model
        if self.cfg.output.save_model:
            save_path = self.output_dir / new_model
            trainer.model.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    def _format_transform(self, example):
        """Format dataset examples into messages format"""
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {'messages': messages}

    def _special_formatting_prompts(self, example):
        """Format using custom template if needed"""
        return f"{self.cfg.dataset.instruction_prompt_template}{example['instruction']}\n{self.cfg.dataset.response_template} {example['output']}"


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point for training with Hydra configuration"""
    print(OmegaConf.to_yaml(cfg))
    trainer = CausalTrainer(cfg)
    trainer.train_sft()


if __name__ == "__main__":
    main()