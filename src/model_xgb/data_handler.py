import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from datetime import datetime as dtime


class DataHandler:
    def __init__(self, dir_root, dir_out, debug=True):
        self.dir_root = dir_root
        self.dir_out = dir_out
        self.debug = debug

        self.dir_images = os.path.join(dir_out, "images")
        self.dir_csv = os.path.join(dir_out, "csv")
        self.dir_models = os.path.join(dir_out, "models")

        self.path_train = os.path.join(self.dir_root, "train.csv")
        self.path_train_labels = os.path.join(self.dir_root, "train_labels.csv")
        # self.path_test = os.path.join(self.dir_root, "test.csv")
        # self.path_submission = os.path.join(self.dir_root, "sample_submission.csv")

        self.df_train = None
        self.df_labels = None
        # self.df_test = None
        # self.df_submission = None

        self.str_ts = dtime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self):
        self.df_train = pd.read_csv(self.path_train)
        self.df_labels = pd.read_csv(self.path_train_labels)
        # self.df_test = pd.read_csv(self.path_test)
        # self.df_submission = pd.read_csv(self.path_submission)

        if self.debug:
            max_rows = 5000
            self.df_train = self.df_train.head(max_rows)
            self.df_labels = self.df_labels.head(max_rows)
            # self.df_test = self.df_test.head(5000)
            # self.df_submission = self.df_submission.head(5000)

    def create_dirs(self):
        for dir_temp in ["images", "csv", "models"]:
            dir_temp = os.path.join(self.dir_out, dir_temp)
            if not os.path.exists(dir_temp):
                os.makedirs(dir_temp)

    def modify_df_labels(self):
        self.df_labels['session_id_num'] = self.df_labels['session_id'].apply(lambda x: int(x.split('_')[0]))
        self.df_labels['question_num'] = self.df_labels['session_id'].apply(lambda x: x.split('_')[-1])

    def print_stats(self):
        print(f"df_train: shape: {self.df_train.shape}")
        print(f"df_labels: shape: {self.df_labels.shape}")

        print(f"df_train: head: {self.df_train.head()}")
        print(f"df_labels: head: {self.df_labels.head()}")

    def get_unique_sessions(self):
        list_uniq_sessions = list(self.df_train['session_id'].unique())
        return list_uniq_sessions

    def get_level_in_session(self, df_sess, level_group):
        cond_lg = df_sess['level_group'] == level_group
        df_lg = df_sess[cond_lg]
        return df_lg

    def get_dict_agg(self, df_session):
        dict_record = {}
        list_level_groups = []

        # Update aggregates -- event counts
        for level_group in list_level_groups:
            df_lg = self.get_level_in_session(df_session, level_group)
            dict_tmp = df_lg['event_name'].value_counts().to_dict()

            # Distinguish same events at different level group
            dict_tmp = {f"{key}:{level_group}": val for key, val in dict_tmp.items}
            dict_record.update(dict_tmp)

        return dict_record

    def get_path_df_agg(self):
        path_df_agg = os.path.join(self.dir_csv, f"df_agg_{self.str_ts}.csv")
        return path_df_agg

    def get_checkpoint_info(self, df_session):
        dict_record = {}
        df_sess_checkpoints = df_session[df_session['event_name'] == "checkpoint"]
        for row_id, row in df_sess_checkpoints.iterrows():
            dict_record.update({
                f'elapsed_time:{row["level_group"]}': row['elapsed_time'],
                f'event_index:{row["level_group"]}': row['index'],
            })

        return dict_record

    def get_question_correctness_labels(self, session_id):
        df_sess_labels = self.df_labels[self.df_labels['session_id_num'] == session_id]

        dict_q_correct = df_sess_labels.set_index('question_num').to_dict()['correct']

        # Additional computation -- get number of correctly answered questions
        dict_q_correct["num_correct"] = sum(dict_q_correct.values())
        return dict_q_correct

    def get_df_agg(self):
        list_uniq_sessions = self.get_unique_sessions()
        len_sessions = len(list_uniq_sessions)
        list_records = []
        for ith_sess, session_id in enumerate(list_uniq_sessions):
            dict_record = {"session_id": session_id}
            cond_session = self.df_train['session_id'] == session_id

            df_session = self.df_train[cond_session]
            print(f"Processing: {ith_sess} of {len_sessions}")

            dict_record.update(self.get_dict_agg(df_session))
            dict_record.update(self.get_checkpoint_info(df_session))
            dict_record.update(self.get_question_correctness_labels(session_id))

            list_records.append(dict_record)

        df_agg = pd.DataFrame.from_records(list_records)
        df_agg.to_csv(self.get_path_df_agg(), index=False)

        return df_agg

    def prepare_dataset(self):
        self.load_data()
        self.create_dirs()
        self.modify_df_labels()
        self.print_stats()


def flow_01():
    dir_root = '../../data/predict-student-performance-from-game-play/'
    dir_out = '../../data/psp_outputs_01/'
    obj_dh = DataHandler(dir_root, dir_out)
    obj_dh.prepare_dataset()


def main():
    flow_01()


if __name__ == '__main__':
    main()
