## python file that generates predictions from a pretrained language model from huggingface transformers
r'''
Run `python generate_data_llama.py ../../data/cladder-v1/cladder-v1-q-balanced.json ../../data/cladder-v1/cladder-v1-meta-models.json`
to get data `causal_benchmark_data_llama.csv` first
'''
import torch
from transformers import  LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import csv
import pandas as pd
from tqdm import trange

def convert_to_norm(value):
    prefix2norm = {
        'Yes': 1,
        'No': 0,
    }
    invalid = -1
    value = str(value).lower().strip().strip('"')

    for prefix, norm in prefix2norm.items():
        if value.startswith(prefix.lower()):
            return norm
    return invalid

def main(locationLlamaHF,outputFileName,inputFileName):
    # Load model directly

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
    # tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',cache_dir="~/cache/")
    # model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',device_map="auto",cache_dir="~/cache/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv(inputFileName)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w', newline='') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(['question_id', 'rung', 'prompt', 'truth', 'truth_norm', 'pred', 'pred_norm','model_version'])  # Include 'question_id' column in the header

    with torch.no_grad():
        for i in trange(df.shape[0]):
            prompt = df.at[i, 'prompt'] + "\nAnswer:"  # Fetch prompt from 'prompt' column
            question_id = df.at[i, 'question_id']  # Fetch question_id from 'question_id' column
            truth = df.at[i, 'truth']
            rung = df.at[i, 'rung']
            truth_norm = df.at[i, 'truth_norm']
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,
                temperature=0
            )
            outputs = tokenizer.decode(output_sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred_norm = convert_to_norm(outputs)
            with open(outputFileName, 'a', newline='') as csvoutput:
                writer = csv.writer(csvoutput)
                writer.writerow([question_id] + [rung] + [prompt] + [truth] + [truth_norm] + [outputs] + [pred_norm] + ['llama007'])



if __name__ == '__main__':
    ## model location HuggingFace format
    locationLlamaHF="./llama-7b"
    outputFileName="./llama007_causal_benchmark_results.csv"
    inputFileName="./causal_benchmark_data_llama.csv"
    main(locationLlamaHF,outputFileName,inputFileName)
