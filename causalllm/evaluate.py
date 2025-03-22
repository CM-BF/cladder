from causalllm.data import DataFile
from causalllm.definitions import ROOT_PATH


class Scorer:
    def __init__(self, files, ask_about, save_perfomance=None):
        if not len(files):
            print('No files for evaluation')
            import sys
            sys.exit()
        self.save_perfomance = save_perfomance if save_perfomance else f'{ROOT_PATH}/outputs/performance.csv'

        from efficiency.log import fread
        data_list = []
        for file in sorted(files):
            data = fread(file) # pd.read_csv(file).to_dict()
            data_list += data
            print(file, len(data))

        import pandas as pd
        df = pd.DataFrame(data_list)
        # df = pd.read_csv(file, index_col=None)
        self.truth_pred_scorer(df, ask_about)

    def apply_score_func(self, df, ask_about, pred_key='pred_norm', truth_key='truth_norm'):
        if ask_about in {'graph'}:
            pred_key = 'pred'
            truth_key = 'truth'

            def score_func(row):
                def txt2edges(txt):
                    txt = txt.replace(' ', '')
                    edges = txt.split(',')
                    edges = {tuple(sorted(i.split('->', 1))) for i in edges}
                    return edges

                def edge_set2node_set(edges):
                    from efficiency.function import flatten_list
                    nodes = flatten_list(edges)
                    return set(nodes)

                from efficiency.function import get_set_f1, get_set_edit_distance

                pred_edges = txt2edges(row[pred_key])
                truth_edges = txt2edges(row[truth_key])
                edge_f1 = get_set_f1(truth_edges, pred_edges)

                pred_nodes = edge_set2node_set(pred_edges)
                truth_nodes = edge_set2node_set(truth_edges)
                node_f1 = get_set_f1(truth_nodes, pred_nodes)

                edit_distance = get_set_edit_distance(truth_edges, pred_edges)
                score_dict = {
                    'node_f1': node_f1,
                    'edge_f1': edge_f1,
                    'edge_edit_distance': edit_distance,
                    'score': edge_f1,
                }
                return score_dict
        else:
            # if ask_about in {'answer', 'query_type'}:
            score_func = lambda row: {'score': row[pred_key] == row[truth_key]}

        # df['score'] = df.apply(score_func, axis=1)
        import pandas as pd
        score_df = df.apply(lambda row: pd.Series(score_func(row)), axis=1)
        df = df.join(score_df)
        print(score_df.mean())
        score_df.describe()

        return df

    def truth_pred_scorer(self, df, ask_about):
        # df.drop(['prompt', 'question_id'], axis=1, inplace=True)

        # - 1. Get scores: Concat calculated scores to the df with customized score function -
        df = self.apply_score_func(df, ask_about)
        # df['score'] = (df['pred_norm'] == df['truth_norm'])

        if ask_about not in {'graph'}:
            from sklearn.metrics import classification_report
            df_valid = df[~df['pred_norm'].isna()]
            for rung in [1, 2, 3]:
                report = classification_report(df_valid[df_valid['rung'] == rung]['truth_norm'],
                                               df_valid[df_valid['rung'] == rung]['pred_norm'], digits=4)
                print(f'Classification report for rung {rung}: \n {report}')
            report = classification_report(df_valid['truth_norm'], df_valid['pred_norm'], digits=4)
            print(f'Classification report for all: \n {report}')


        res_dfs = []
        # -- 2. Results grouping by model_version --
        for uniq_vign_key in ['model_version']:
            try:
                res_df = self._res_by_group(df, uniq_vign_key)
                res_dfs.append(res_df)
            except:
                continue
        for model_version in sorted(df['model_version'].unique().tolist()):
            new_df = df[df['model_version'] == model_version]; print('-' * 20, model_version, '-' * 20)
            # - 3. In each model version, group different properties like sensical, query_type, rung, phenomenon, and simpson. -
            for uniq_vign_key in DataFile.metadata_keys:
                try:
                    res_df = self._res_by_group(new_df, uniq_vign_key)
                    res_df['model_version'] = model_version
                    res_dfs.append(res_df)
                except:
                    continue

        import pandas as pd
        res_df = pd.concat(res_dfs) # - a weird way to show the performance -
        print(res_df)
        res_df.to_csv(self.save_perfomance)

        # -- 4. Aggregate the results by query_type x model_version | score --
        def pivot_df(df, rows='query_type', columns='model_version', score_col='score', verbose=True):
            pivot_df = df.pivot_table(index=rows, columns=columns, values=score_col, aggfunc='first')

            pivot_df.reset_index(inplace=True)
            pivot_df.fillna('---', inplace=True)
            pivot_df.columns.name = None

            desired_order = sorted(df[rows].apply(str).unique().tolist())
            pivot_df.set_index(rows, inplace=True)
            pivot_df = pivot_df.reindex(desired_order)
            pivot_df.reset_index(inplace=True)
            if verbose: print(pivot_df)
            return pivot_df
        pivot_df(res_df)

    @staticmethod
    def _res_by_group(df, uniq_vign_key, result_key='score', return_obj=['group_dict', 'consistency_rate'][0]):
        # Group by 'group' column and count the occurrences of each value in the 'result' column
        g = df.groupby(uniq_vign_key)[result_key]
        dff = round(g.mean() * 100, 2).reset_index()
        dff['count'] = g.count().to_list()
        print(dff)
        return dff

        g_counts = df.groupby(uniq_vign_key)[result_key].value_counts()
        g_counts.name = 'performance'  # otherwise, there will be an error saying that `result_key` is used
        # for both the name of the pd.Series object, and a column name
        g_totals = g_counts.groupby(uniq_vign_key).sum()
        g_perc = round(g_counts / g_totals * 100, 2)
        g_major = g_perc.groupby(uniq_vign_key).max()
        consistency_rate = round(g_major.mean(), 2)

        if return_obj == 'group_dict':
            g_perc_clean = g_perc.drop([False],
                                       level=result_key, errors='ignore')
            # dff = g_perc_clean.reset_index() # turn into df
            # g_perc_clean.to_csv(performance_file)

            print(g_perc_clean)
            # print('[Info] The above results are saved to', performance_file)

            return g_perc_clean.to_dict()
        elif return_obj == 'consistency_rate':
            return consistency_rate
