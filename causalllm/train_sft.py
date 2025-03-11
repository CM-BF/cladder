import operator
import os.path
from functools import partial
from pathlib import Path
from typing import Literal, Any, List, TypedDict, Annotated, Dict, Tuple
import re

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
        if self.cfg.testing.only_test:
            assert self.cfg.testing.test_model_path is not None
            if Path(self.cfg.testing.test_model_path).exists():
                self.output_dir = Path(self.cfg.testing.test_model_path).parent
                self.cfg = self.load_config_with_testing_override(self.output_dir / "config_dump.yaml", self.cfg.testing)
            else:
                self.output_dir = Path.cwd()
        else:
            self.output_dir = Path.cwd()  # Hydra changes working directory
            # Save the config for reproducibility
            with open(self.output_dir / "config_dump.yaml", "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
        print(OmegaConf.to_yaml(self.cfg))
        print(f"Working directory: {self.output_dir}")

    def train_sft(self):
        """Train a model using Supervised Fine-Tuning with configurations from Hydra"""
        # Extract config values
        model_name = self.cfg.model.name
        new_model = self.cfg.model.new_model
        enable_cot = self.cfg.experiment.enable_cot
        enable_fewshot = self.cfg.experiment.enable_fewshot
        given_cot_until_step = self.cfg.experiment.given_cot_until_step

        # Setup output files
        write_out_file = self.output_dir / 'output.csv'

        # Create text interface for prompt composition
        text_interface = TextInterfaceForLLMs(
            str(write_out_file),
            ask_about=self.cfg.experiment.ask_about,
            enable_fewshot=enable_fewshot,
            enable_cot=enable_cot,
            given_cot_until_step=given_cot_until_step
        )

        # Load dataset


        # Run testing if enabled
        if self.cfg.testing.only_test:
            self.test_model(model_name, new_model, text_interface)
        else:
            # Process each dataset file
            self._perform_sft(model_name, text_interface, new_model)
            self.test_model(model_name, new_model, text_interface)



    def _perform_sft(self, model_name, text_interface, new_model):
        """Perform the actual SFT training for a given dataset"""
        # Extract configurations
        lora_config = self.cfg.lora
        quant_config = self.cfg.quantization
        training_config = self.cfg.training
        dataset_config = self.cfg.dataset

        # Setup device map
        device_map = {"": 0} if training_config.device_map == 0 else training_config.device_map

        # Prepare the dataset
        data_file_obj = DataFileList(data_name=dataset_config.train_data, ask_about=self.cfg.experiment.ask_about).data_objs[0]
        train_dataset, eval_dataset = self._prepare_data(data_file_obj, text_interface)

        train_dataset, eval_dataset = map(partial(self.format_dataset, dataset_config=dataset_config), [train_dataset, eval_dataset])

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
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
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
            tokenizer.save_pretrained(save_path)

    def format_dataset(self, dataset, dataset_config):
        # "prompt" is a special column key for SFTTrainer's data preprocessing
        dataset = dataset.rename_columns({"prompt": "instruction", "response": "output"})
        other_columns = [i for i in dataset.column_names if i not in ["instruction", "output"]]
        dataset = dataset.remove_columns(other_columns)
        # Format dataset for model input
        if not dataset_config.use_special_template:
            dataset = dataset.map(self._format_transform, desc="Formatting dataset to ChatML")
        return dataset

    def test_model(self, model_name, new_model_name, text_interface):
        """Test the trained model on the test dataset and compute accuracy"""
        print("\n=== Starting model testing ===")

        # Extract test config
        test_config = self.cfg.testing

        # Determine model path
        if test_config.only_test:
            model_path = test_config.test_model_path
        else:
            model_path = self.output_dir / new_model_name


        # Load the model
        device_map = "auto" if torch.cuda.is_available() else None
        print(f"Loading model from {model_path}")

        # Setup quantization if needed
        if test_config.use_quantization:
            quant_config = self.cfg.quantization
            compute_dtype = getattr(torch, quant_config.compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.load_in_4bit,
                bnb_4bit_quant_type=quant_config.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_config.use_double_quant,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=self.cfg.model.trust_remote_code,
                # attn_implementation = "flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=self.cfg.model.trust_remote_code
            )
        try:
            print(f"Active adapter: {model.active_adapter()}")
        except:
            print("No active adapter: Base model")
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.cfg.model.trust_remote_code
            )
        except:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.cfg.model.trust_remote_code
            )
        tokenizer.padding_side = 'left'

        # Process each dataset file for testing
        # Prepare test data
        data_file_obj = \
        DataFileList(data_name=self.cfg.testing.test_data, ask_about=self.cfg.experiment.ask_about).data_objs[0]
        _, test_dataset = self._prepare_data(data_file_obj, text_interface)
        test_data = test_dataset.to_list()

        batch_size = test_config.test_batch_size

        for i in tqdm(range(0, len(test_data), batch_size), desc="Testing batches"):
            batch = test_data[i:i + batch_size]
            batch_queries = [item["prompt"] for item in batch]

            # Generate responses for the batch
            batch_responses = self._generate_batch_responses(
                model, tokenizer, batch_queries, test_config
            )

            # Extract answers from responses
            for j, response in enumerate(batch_responses):
                pred = self._extract_answer(response)
                test_data[i + j]['pred'] = pred

            # Print sample for debugging
            if i % (5 * batch_size) == 0 and batch:
                print(
                    f"Sample Query: {batch_queries[0]}\n\nResponse: {batch_responses[0]}\n\nExtracted Answer: {test_data[i]['pred']}\n\nExpected Answer: {test_data[i]['truth']}\n\n")

        # for i, datum in tqdm(enumerate(test_data), desc="Testing"):
        #     query = datum["prompt"]
        #
        #     # Generate response using the model
        #     response = self._generate_model_response(model, tokenizer, query, test_config)
        #
        #     # Extract the answer
        #     pred = self._extract_answer(response)
        #     datum['pred'] = pred
        #     if i % 100 == 0:
        #         print(f"Query: {query}\n\nResponse: {response}\n\nExtracted Answer: {pred}\n\nExpected Answer: {datum['truth']}\n\n")

        test_df = pd.DataFrame(test_data)
        test_df.to_csv(text_interface.save_path, index=False)

        text_interface.response_processor(model_version=f"{new_model_name}")

        scorer = Scorer([text_interface.save_path], ask_about='answer', save_perfomance=text_interface.save_path)

    def _prepare_data(self, data_file_obj, text_interface):
        """Prepare data for testing"""
        # Use the same preprocessing as for training
        text_interface.prepare_prompt_sft(data_file_obj.data, reasoning=self.cfg.experiment.reasoning)
        data = text_interface.data_in
        all_samples = list(map(lambda x: {k: v for k, v in x.items() if k != 'old'}, data))

        # Filter based on test split if needed
        dataset = Dataset.from_list(all_samples)
        split_dataset = dataset.train_test_split(
            train_size=self.cfg.dataset.percent_of_train_dataset,
            seed=self.cfg.dataset.seed,
            shuffle=self.cfg.dataset.shuffle
        )
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]

        return train_dataset, test_dataset

    def _generate_model_response(self, model, tokenizer, instruction, test_config):
        """Generate a response from the model given an instruction"""
        # Format the instruction if needed
        if test_config.use_chat_template:
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = instruction

        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=test_config.max_new_tokens,
                temperature=test_config.temperature,
                top_p=test_config.top_p,
                do_sample=test_config.do_sample,
                num_beams=test_config.num_beams,
                repetition_penalty=test_config.repetition_penalty,
            )

        # Extract the new tokens only (excluding the input)
        output_ids = output_ids[:, inputs.input_ids.shape[1]:]

        # Decode the tokens to get the response
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return response

    def _generate_batch_responses(self, model, tokenizer, instructions, test_config):
        """Generate responses for a batch of instructions"""

        # Format the instructions if needed
        if test_config.use_chat_template:
            input_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for instruction in instructions
            ]
        else:
            input_texts = instructions

        # Tokenize all inputs
        batch_inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Add a reasonable max length to avoid OOM
        ).to(model.device)

        # Define stopping criteria for </answer> tag
        stopping_criteria = None
        if hasattr(test_config, "use_stopping_criteria") and test_config.use_stopping_criteria:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class AnswerTagStoppingCriteria(StoppingCriteria):
                def __init__(self, tokenizer, stop_strings=["</answer>"]):
                    self.tokenizer = tokenizer
                    self.stop_ids = [
                        self.tokenizer.encode(stop_str, add_special_tokens=False)
                        for stop_str in stop_strings
                    ]

                def __call__(self, input_ids, scores, **kwargs):
                    # Check last generated tokens for stop sequences
                    for batch_idx, seq in enumerate(input_ids):
                        seq_len = len(seq)
                        for stop_ids in self.stop_ids:
                            stop_len = len(stop_ids)
                            if seq_len >= stop_len:
                                if seq[-stop_len:].tolist() == stop_ids:
                                    return True
                    return False

            stopping_criteria = StoppingCriteriaList([
                AnswerTagStoppingCriteria(tokenizer)
            ])

        # Generate responses
        with torch.no_grad():
            output_ids = model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_new_tokens=test_config.max_new_tokens,
                temperature=test_config.temperature,
                top_p=test_config.top_p,
                do_sample=test_config.do_sample,
                num_beams=test_config.num_beams,
                repetition_penalty=test_config.repetition_penalty,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Extract only the newly generated tokens for each sequence
        responses = []
        for i, output in enumerate(output_ids):
            input_length = batch_inputs.input_ids[i].shape[0]
            response_ids = output[input_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)

        return responses

    def _extract_answer(self, response):
        """Extract 'Yes' or 'No' from between <answer> tags"""
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            answer_text = match.group(1).strip().lower()
            # Check if the answer is 'yes' or 'no'
            if 'yes' in answer_text:
                return 'yes'
            elif 'no' in answer_text:
                return 'no'
        else:
            # if cannot find the <answer> tags, find the first yes or no, neglecting capitalization
            answer_text = re.search(r'\b(yes|no)\b', response, re.IGNORECASE)
            if answer_text:
                return answer_text.group(1).lower()

        return None

    def _compare_answer(self, extracted_answer, truth):
        """Compare the extracted answer with the truth value"""
        # Normalize both to lowercase
        extracted_lower = extracted_answer.lower() if extracted_answer else None
        truth_lower = truth.lower() if truth else None

        # Compare
        return extracted_lower == truth_lower

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

    def load_config_with_testing_override(self, config_file: Path, testing_config = None) -> DictConfig:
        """
        Load a saved config file and optionally override its testing section.
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Convert to OmegaConf
        cfg = OmegaConf.load(config_file)

        # Override testing config if provided
        if testing_config is not None:
            cfg.testing = testing_config

        return cfg


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Main entry point for training with Hydra configuration"""
    trainer = CausalTrainer(cfg)
    trainer.train_sft()


if __name__ == "__main__":
    main()