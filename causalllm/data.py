from causalllm.definitions import ROOT_PATH, paraphrase_i


class DataFileList:
    def __init__(self, data_name = 'cladder-v1/cladder-v1-q-balanced', shuffle=True, ask_about=None):
        from glob import glob

        file_pattern = f'{ROOT_PATH}/data/{data_name}.json'

        data_files = sorted(glob(file_pattern))
        print('Starting to get data from these files:', data_files)
        if not len(data_files):
            print('There are no files in your specified path:', file_pattern)
            import sys
            sys.exit()
        data_objs = []
        for file in data_files:
            data_obj = DataFile(file, shuffle=shuffle, ask_about=ask_about)
            data_objs.append(data_obj)
        self.data_objs = data_objs


class DataFile:
    metadata_keys = ['sensical', 'query_type', 'rung', 'phenomenon', 'simpson']

    def __init__(self, file, shuffle=True, len=None, ask_about=None):
        self.read_in_file = file
        self.file_shuffled = file.replace('.json', '_rand.json')

        self.data = self.load_data(file, shuffle=shuffle, ask_about=ask_about)
        self.data = self.data[:len]

    def get_ids_to_exclude(self, file=f'{ROOT_PATH}/data/updated_data_ids_not_ready.txt'):
        from efficiency.log import fread
        data = '\n'.join(fread(file))
        ids = data.split(', ')
        ids = [int(i) for i in ids if i]
        return ids

    def paraphrase(self, datum):
        paraphrases = [
            "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships",
            "Think of a self-contained, hypothetical setting with just the specified conditions, and devoid of any unknown factors or causal connections",
            "Consider a self-contained, hypothetical world with solely the mentioned conditions, and is free from any hidden factors or cause-and-effect relationships",
            "Imagine a self-contained, hypothetical setting with merely the stated conditions, and absent any unmentioned factors or causative links",
            "Think of a self-contained, hypothetical world with only the given conditions, and is void of any unknown factors or causative relationships",
        ]
        if 'given_info_para' in datum:
            datum['given_info'] = [datum['given_info'], datum['given_info_para']][paraphrase_i % 2]  # TODO: add "para"
        datum['background'] = datum['background'].replace(paraphrases[0], paraphrases[paraphrase_i])
        return datum

    def load_data(self, file, inspect_data=False, exclude_query_types={}, shuffle=True, ask_about=None):
        from efficiency.log import fread, show_var, fwrite, print_df_value_count
        from efficiency.function import random_sample, search_nested_dict

        models_data_name = 'cladder-v1/cladder-v1-meta-models'
        with open(f'{ROOT_PATH}/data/{models_data_name}.json', 'r') as f:
            import json

            meta_models = json.load(f)

        # --- Load raw data ---
        if not shuffle: data = fread(file)
        if shuffle:
            import os
            if os.path.isfile(self.file_shuffled):
                data = fread(self.file_shuffled)
            else:
                data = fread(file)
                data = random_sample(data, size=None)
                import json
                fwrite(json.dumps(data, indent=4), self.file_shuffled, verbose=True)

        datum2raw_prompt = lambda datum: f"{datum['background'].strip()}" \
                                         f" {datum['given_info'].strip()}" \
                                         f" {datum['question'].strip()}".replace(' ', ' ')
        datum2raw_prompt_without_q = lambda datum: f"{datum['background'].strip()}" \
                                                   f" {datum['given_info'].strip()}".replace(' ', ' ')
        ids_to_exclude = self.get_ids_to_exclude()
        show_var(['len(data)'])
        data = [i for i in data if i['question_id'] not in ids_to_exclude]
        show_var(['len(data)'])

        # --- Processing ---
        new_data = []
        for datum in data:
            model = meta_models[datum['meta']['model_id']]
            datum['background'] = model['background']
            datum['variable_mapping'] = model['variable_mapping']
            datum['sensical'] = 1
            if model.get('nonsense', False):
                datum['sensical'] = 0
            if model.get('anticommonsense', None) is not None:
                datum['sensical'] = -1
            datum['meta']['simpson'] = model.get('simpson', False)
            datum['meta']['story_id'] = datum['meta']['story_id']
            datum['meta']['phenomenon'] = datum['meta']['graph_id']
            datum['meta']['query'] = {'query_type': datum['meta']['query_type'],
                                      'rung': datum['meta']['rung']}

            datum = self.paraphrase(datum)
            new_datum = {'old': datum} # --- 'old' means the raw data ---

            # --- Extract some target data ---
            for k in ["question_id", "descriptive_id", ] + self.metadata_keys:
                v = search_nested_dict(datum, k)
                if v is not None:
                    new_datum[k] = search_nested_dict(datum, k)
            new_datum.update({
                'raw_prompt': datum2raw_prompt(datum),
                'raw_prompt_without_q': datum2raw_prompt_without_q(datum),
                'truth': search_nested_dict(datum, ask_about),
            })
            # new_datum.update({k: v for k, v in datum.items() if k in {"background", "given_info", "question", }})
            new_data.append(new_datum)
        data = new_data
        if file.endswith('prop_cna_100_each.json'):
            data = [i for i in data if i['query_type'] in {'cou'}][:100]

        data = [i for i in data if i['query_type'] not in exclude_query_types]

        if inspect_data:
            import pandas as pd

            df = pd.DataFrame(data)
            columns = (set(self.metadata_keys) & df.columns) | {'truth', }
            print_df_value_count(df, columns)
        return data
