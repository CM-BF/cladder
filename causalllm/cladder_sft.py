import operator
from pathlib import Path
from typing import Literal, Any, List, TypedDict, Annotated, Dict, Tuple

import asyncio

from langchain.chains.question_answering.map_reduce_prompt import messages
from pydantic import BaseModel
from sqlalchemy.dialects.mysql.mariadb import loader
from sympy.assumptions.satask import satask
from tqdm import tqdm
import pandas as pd
from itertools import product
from datasets import Dataset
import networkx as nx
from tqdm.asyncio import tqdm_asyncio

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig

from causalllm.data import DataFileList
from causalllm.evaluate import Scorer
from causalllm.prompt_utils import partial_replace, TextInterfaceForLLMs
from causalllm.definitions import ROOT_PATH, missing_step, paraphrase_i
from causalllm.structured_data_template import MultiAgentData, SCGString, Verifier, Reflector
from causalllm.graph_parser import parse_graph_from_text, draw_graph
from causalllm.langgraph_builder import AgentState, ConfigSchema, executor
from graph_parser import visualize_langgraph
from langgraph_builder import models
from jsonargparse import CLI

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

import openai

from trl import SFTTrainer
from datasets import load_dataset


class Tester:

    def __init__(self):
        from efficiency.function import set_seed
        set_seed()

    def run_default_test(self, just_scoring: bool = False, enable_cot: bool = False,
                         enable_fewshot: bool = False, model_versions: List[str] = [],
                         given_cot_until_step: int = None,
                         ask_about: Literal[
                             'answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate'] = 'answer',
                         k_graph: int = 5,
                         max_scg_trials: int = 5,
                         max_num_executions: int = 10,
                         reasoning: bool = False, ):
        """
        Args:
            just_scoring: if True, the function will only score the existing responses in the files
            enable_cot: if True, the function will guide the model to answer the question step by step
            enable_fewshot: if True, the function will provide the model with a few-shot example before asking the question
            model_versions: a list of model versions to be tested
            ask_about: the type of question to be asked: 'answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate'
            given_cot_until_step: = [None, 1, 2, 3, 4]
            k_graph: the number of graph design coroutines
            max_scg_trials: the maximum number of trials for single SCG design
            max_num_executions: the maximum number of execution trials
            reasoning: if True, the function will use reasoning to answer the question
        Returns:
        """
        assert given_cot_until_step in [None, 1, 2, 3, 4, 5, 6]

        system_prompt = '''
You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content.
        '''.strip()
        max_tokens = 200
        if ask_about == 'answer':
            max_tokens = 1
        elif ask_about == 'query_type':
            max_tokens = 20

        ask_about_suffix = f'_{ask_about.replace("_", "-")}' if ask_about != 'answer' else ''
        missing_step_suffix = f'_no{missing_step.replace("_", "-")}' if missing_step else ''

        write_out_files = []

        if just_scoring:
            combs = list(product(model_versions, [False], range(0, 5))) \
                    + list(product(['gpt4'], [True], range(0, 5)))
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        else:
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        print(combs)
        # if not just_scoring: import pdb;pdb.set_trace()
        for model_version, enable_cot, para_i in combs:

            if 'gpt' not in model_version:
                max_tokens += 2

            # == make file name ==
            cot_suffix = 'cot' if enable_cot else ''
            if cot_suffix:
                cot_suffix += str(given_cot_until_step) if given_cot_until_step is not None else ''
            fewshot_suffix = '_10shot' if enable_fewshot else ''
            para_suffix = f'_pa{para_i}' if para_i else ''

            from datetime import datetime
            START_TIME = datetime.now().strftime("%Y%m%d%H%M%S")

            exp_folder = Path(f'{ROOT_PATH}/outputs/{model_version}{cot_suffix}' \
                              f'{fewshot_suffix}{ask_about_suffix}{missing_step_suffix}{para_suffix}{START_TIME}')
            exp_folder.mkdir(parents=True, exist_ok=True)

            write_out_file = exp_folder / 'output.csv'
            write_out_files.append(str(write_out_file))
            # == make file name end ==

            # --- Create LLM Text interface taking charge of template/prompt composer/response processor ---
            text_interface = TextInterfaceForLLMs(write_out_file, ask_about=ask_about, enable_fewshot=enable_fewshot,
                                                  enable_cot=enable_cot, given_cot_until_step=given_cot_until_step)

            # graph, multi_agent_data = scg_design(just_scoring, 'gpt-4o')
            # capybara = load_dataset('trl-lib/Capybara', split='train')
            # cladder = load_dataset('causal-nlp/CLadder', split='full_v1.5_default')

            model_name = 'Qwen/Qwen2.5-3B-Instruct'

            df_list = DataFileList(ask_about=ask_about)
            for data_file_obj in df_list.data_objs:
                if not just_scoring:
                    sft(data_file_obj, model_name, reasoning, text_interface)

                text_interface.response_processor(model_version=f"{model_version}{cot_suffix}")

        scorer = Scorer(write_out_files, ask_about)

    def train_sft(self, just_scoring: bool = False, enable_cot: bool = False,
                  enable_fewshot: bool = False, model_versions: List[str] = [],
                  given_cot_until_step: int = None,
                  ask_about: Literal[
                      'answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate'] = 'answer',
                  model_name: str = 'Qwen/Qwen2.5-3B-Instruct',
                  new_model="QWen-CLadder-agent",
                  reasoning: bool = False, ):
        """
        Args:
            just_scoring: if True, the function will only score the existing responses in the files
            enable_cot: if True, the function will guide the model to answer the question step by step
            enable_fewshot: if True, the function will provide the model with a few-shot example before asking the question
            model_versions: a list of model versions to be tested
            ask_about: the type of question to be asked: 'answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate'
            given_cot_until_step: = [None, 1, 2, 3, 4]
            k_graph: the number of graph design coroutines
            max_scg_trials: the maximum number of trials for single SCG design
            max_num_executions: the maximum number of execution trials
            reasoning: if True, the function will use reasoning to answer the question
        Returns:
        """
        assert given_cot_until_step in [None, 1, 2, 3, 4, 5, 6]

        write_out_files = []

        from datetime import datetime
        START_TIME = datetime.now().strftime("%Y%m%d%H%M%S")

        exp_folder = Path(f'{ROOT_PATH}/outputs/{model_name}{"_reasoning" if reasoning else ""}_{START_TIME}')
        exp_folder.mkdir(parents=True, exist_ok=True)

        write_out_file = exp_folder / 'output.csv'
        write_out_files.append(str(write_out_file))
        # == make file name end ==

        # --- Create LLM Text interface taking charge of template/prompt composer/response processor ---
        text_interface = TextInterfaceForLLMs(write_out_file, ask_about=ask_about,
                                              enable_fewshot=enable_fewshot,
                                              enable_cot=enable_cot, given_cot_until_step=given_cot_until_step)

        df_list = DataFileList(ask_about=ask_about)
        for data_file_obj in df_list.data_objs:
            sft(data_file_obj, model_name, reasoning, text_interface, exp_folder, new_model)


def sft(data_file_obj, model_name, reasoning, text_interface, exp_folder, new_model):
    # LoRA parameters - adjusted for Qwen architecture
    lora_r = 64
    lora_alpha = lora_r * 2
    lora_dropout = 0.1
    # Updated target modules for Qwen architecture
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # QLoRA parameters
    load_in_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_use_double_quant = False

    # TrainingArguments parameters
    num_train_epochs = 1
    fp16 = False
    bf16 = True  # Changed to True for better performance with Qwen models
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
    use_special_template = False
    response_template = " ### Answer:"
    instruction_prompt_template = '"### Human:"'
    use_llama_like_model = False  # Changed to False since we're using Qwen

    # Load dataset (you can process it here)
    def format_transform(example):
        messages = [
            {"role": "user", "content": example["instruction"]},  # Changed 'system' to 'user' for Qwen
            {"role": "assistant", "content": example["output"]},
        ]
        return {'messages': messages}

    text_interface.prepare_prompt_sft(data_file_obj.data, reasoning=reasoning)
    data = text_interface.data_in
    all_samples = list(map(lambda x: {k: v for k, v in x.items() if k != 'old'}, data))
    dataset = Dataset.from_list(all_samples)

    percent_of_train_dataset = 0.9
    # "prompt" is a special column key for SFTTrainer's data preprocessing, avoid using it unless you know what you're doing
    dataset = dataset.rename_columns({"prompt": "instruction", "response": "output"})
    other_columns = [i for i in dataset.column_names if i not in ["instruction", "output"]]
    dataset = dataset.remove_columns(other_columns)
    if not use_special_template:
        dataset = dataset.map(format_transform, desc="Formatting dataset to ChatML for Qwen-Instruct")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True  # Required for Qwen models
    )
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    # No need to set custom chat_template as Qwen models have their own template

    def special_formatting_prompts(example):
        # Format using Qwen's expected formatting
        return f"{instruction_prompt_template}{example['instruction']}\n{response_template} {example['output']}"

    if use_special_template:
        formatting_func = special_formatting_prompts
        response_template_tokens = tokenizer.encode(response_template, add_special_tokens=False)
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_tokens, tokenizer=tokenizer)
    else:
        formatting_func = None
        collator = None  # Let SFTTrainer create a default collator

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
    trainer.model.save_pretrained(exp_folder / new_model)


if __name__ == '__main__':
    CLI(Tester, parser_mode='omegaconf')
