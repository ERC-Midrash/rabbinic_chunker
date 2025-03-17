""" E2E flow for testing the chunking hypotheses. Generates the Precision, Recall, F1 scores """

import re
import pandas as pd
from pathlib import Path

DELIM = '//'
BREAK_MARK = 'B'
SEMICOLON_MARK = 'P'
MAYBE_MARK = 'M'
MEANINGLESS_MARKS = ('O', 'C', 'N')  ##  open and close parenthesis, and never chunk marking which we don't use yet

ALL_MARKS = BREAK_MARK + SEMICOLON_MARK + MAYBE_MARK  # all possible tags that could appear in the ref.

COMBOS = [
    BREAK_MARK,
    BREAK_MARK + SEMICOLON_MARK,
    ALL_MARKS
]


def handle_uncertain_tagging(gt_text, policy='strict'):
    """ where the tagged text has two marks, which to choose? """
    pattern_1 = f'{SEMICOLON_MARK+MAYBE_MARK}|{MAYBE_MARK+SEMICOLON_MARK}'
    pattern_2 = f'{SEMICOLON_MARK+BREAK_MARK}|{BREAK_MARK+SEMICOLON_MARK}'
    pattern_3 = f'{MAYBE_MARK+BREAK_MARK}|{BREAK_MARK+MAYBE_MARK}'
    if policy == 'strict':
        gt_text = re.sub(pattern=pattern_1, repl=SEMICOLON_MARK, string=gt_text)
        gt_text = re.sub(pattern=pattern_2, repl=BREAK_MARK, string=gt_text)
        gt_text = re.sub(pattern=pattern_3, repl=BREAK_MARK, string=gt_text)
    if policy == 'relaxed':
        gt_text = re.sub(pattern=pattern_1, repl=MAYBE_MARK, string=gt_text)
        gt_text = re.sub(pattern=pattern_2, repl=SEMICOLON_MARK, string=gt_text)
        gt_text = re.sub(pattern=pattern_3, repl=MAYBE_MARK, string=gt_text)
    return gt_text


def remove_meaningless(gt_text):
    for mark in MEANINGLESS_MARKS:
        gt_text = gt_text.replace(mark, '')
    return gt_text


def process_all(path, id_col, cite_col, hyp_col, ref_col, metadata_cols, join=True):
    """ top level method for this module """
    df = pd.read_csv(path, encoding='utf-8_sig')

    full_result_df = (
        df.groupby(metadata_cols)
        .apply(lambda x: process_chunking_table(x, id_col, cite_col, hyp_col, ref_col, metadata_cols))
        .reset_index(drop=True)
    )

    if join:
        full_result_df = pd.merge(left=df, right=full_result_df, left_on=[id_col]+metadata_cols, right_on=[id_col]+metadata_cols)

    return full_result_df


def process_chunking_table(text_df, id_col, cite_col, hyp_col, ref_col, metadata_cols):
    """
    like the man said - process the chunking table
    :param text_df: dataframe with the data to analyze
    :param cite_col: name of column with citation
    :param hyp_col: name of column with suggested chunking
    :param ref_col: name of column with ground truth
    :return: accuracy scores
    """
    toolname = text_df.iloc[0][metadata_cols[0]]
    print(f"\n*** Analyzing data for tool: {toolname}")
    dataset = []
    all_tool_errors = []
    for _, row in text_df.iterrows():
        dataset.append((row[id_col], row[cite_col], row[hyp_col], row[ref_col]))
    scores_df, error_list = evaluate_chunkings(dataset, toolname)
    all_tool_errors += error_list
    for col in metadata_cols:
        scores_df[col] = text_df[col].iloc[0]
    print(f"Here are the {len(all_tool_errors)} problems we see so far:")
    print(all_tool_errors)
    return scores_df


def evaluate_chunkings(chunkings, toolname):
    """
    eval those chunkings!
    :param chunkings: list of tuples to evaluate
    :return:
    """
    result = []
    error_list = []
    for source_id, source, hyp, ref in chunkings:
        # print(f"{toolname}: Processing source {source_id} ({source})")
        try:
            evaluation_dict = evaluate_chunking(hyp, ref)['scores']
            res_dict = {'source_id': source_id, 'source': source}
            for tags in COMBOS:
                for score in ['precision', 'recall', 'f1']:
                    res_dict[f'{tags}_{score}'] = evaluation_dict[tags][score]
            result.append(res_dict)
        except ValueError as ve:
            error_list.append({'toolname': toolname, 'source_id': source_id})

    return pd.DataFrame(result), error_list


def remove_newlines(text):
    """ remove newlines from text """
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text


def evaluate_chunking(hyp, ref):
    """ compare hyp to ref and score it """
    hyp = remove_newlines(hyp)
    ref = remove_meaningless(ref)
    ref = handle_uncertain_tagging(ref)

    result = {'stats': count_stuff(hyp, ref), 'scores': {}}
    for tags in COMBOS:
        tag_scores = get_tag_accuracy_scores(hyp, ref, tags)
        result['scores'][tags] = tag_scores
    return result


def get_tag_accuracy_scores(hyp, ref, tags):
    """
    Get precision, recall, f1 scores, depending on the tags in the ref we want to consider.
    :param hyp:
    :param ref:
    :param tags:
    :return:
    """

    hyp_idx = 0
    ref_idx = 0
    hyp_words = hyp.split()
    ref_words = ref.split()
    TP = 0
    FP = 0
    FN = 0
    while hyp_idx < len(hyp_words) and ref_idx < len(ref_words):
        # print(f"\thyp={hyp_words[hyp_idx]} vs. ref={ref_words[ref_idx]}")
        if hyp_words[hyp_idx] == ref_words[ref_idx]:
            hyp_idx += 1
            ref_idx += 1

        elif hyp_words[hyp_idx] == DELIM and ref_words[ref_idx] in tags:  # match!
            TP += 1
            hyp_idx += 1
            ref_idx += 1

        elif hyp_words[hyp_idx] != DELIM and ref_words[ref_idx] in tags:  # missed delimiter
            FN += 1
            ref_idx += 1

        elif hyp_words[hyp_idx] == DELIM and ref_words[ref_idx] not in tags:  # incorrect delimiter
            FP += 1
            hyp_idx += 1

        elif hyp_words[hyp_idx] != DELIM and ref_words[ref_idx] in ALL_MARKS + ''.join(MEANINGLESS_MARKS):  # irrelevant tag - skip
            ref_idx += 1

        else:
            # print(f"This should not happen. Ever. hyp_words[{hyp_idx}] = {hyp_words[hyp_idx]}, ref_words[{ref_idx}] = {ref_words[ref_idx]}")
            error_str = f"This should not happen. Ever. hyp_words[{hyp_idx}] = {hyp_words[hyp_idx]}, ref_words[{ref_idx}] = {ref_words[ref_idx]}"
            error_str += f"\nhyp = {' '.join(hyp_words[hyp_idx-10:hyp_idx+10])}\nref = {' '.join(ref_words[ref_idx-15:ref_idx+15])}"

            raise ValueError(error_str)

    precision = round(float(TP) / (TP + FP), 3) if TP + FP > 0 else None
    recall = round(float(TP) / (TP + FN), 3) if TP + FN > 0 else None
    if recall is None or precision is None or precision + recall == 0.0:
        f1 = None
    else:
        f1 = round((2 * precision * recall) / (precision + recall), 3)
    tag_scores = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    return tag_scores


def add_specialized_cross_f1(df, precision_tag, recall_tag, new_col):
    def f1(row):
        p = row[f'{precision_tag}_precision']
        r = row[f'{recall_tag}_recall']
        return round(2*(p*r)/(p+r), 3)

    df[new_col] = df.apply(f1, axis=1)
    return df


def count_stuff(hyp, ref):
    """ there is stuff and we want to count it. """
    result = {}
    result['hyp_chunks'] = len(hyp.split(DELIM))
    result[f'ref_{BREAK_MARK}_chunks'] = len(ref.split(BREAK_MARK))
    result[f'ref_{BREAK_MARK+SEMICOLON_MARK}_chunks'] = len(re.split(string=ref, pattern=rf'{BREAK_MARK}|{SEMICOLON_MARK}'))
    result[f'ref_{ALL_MARKS}_chunks'] = len(re.split(string=ref, pattern=rf'{BREAK_MARK}|{SEMICOLON_MARK}|{MAYBE_MARK}'))
    result[f'ref_{BREAK_MARK}_marks'] = len(re.findall(pattern=BREAK_MARK, string=ref))
    result[f'ref_{SEMICOLON_MARK}_marks'] = len(re.findall(pattern=SEMICOLON_MARK, string=ref))
    result[f'ref_{MAYBE_MARK}_marks'] = len(re.findall(pattern=MAYBE_MARK, string=ref))
    return result


def combine_gt_with_hyps(gt_path, hyps_path, target_path, source_ids=None):
    def add_space(text):
        """ This is for cases where we have multiple english letter markings """
        pattern = r'([^A-Z\s])([A-Z])'
        spaced_text = re.sub(pattern=pattern, repl=r'\1 \2', string=text, flags=re.DOTALL)
        return spaced_text

    gt_df = pd.read_csv(gt_path, encoding='utf-8_sig')
    gt_df['ground_truth'] = gt_df['ground_truth'].apply(add_space)

    hyp_df = pd.read_csv(hyps_path, encoding='utf-8_sig')
    hyp_df.rename(columns={'chunked_text': 'hyp'}, inplace=True)
    hyp_df = hyp_df[hyp_df['use in paper?'] == True]
    if source_ids:
        hyp_df = hyp_df[hyp_df['text_id'].isin(source_ids)]

    # join them on the ID column
    combined_df = pd.merge(left=gt_df, right=hyp_df, left_on='source_id', right_on='text_id')
    combined_df = combined_df[['source_id', 'source', 'ground_truth', 'hyp', 'name']]

    combined_df['M_count'] = combined_df['ground_truth'].str.count('M')
    combined_df['B_count'] = combined_df['ground_truth'].str.count('B')
    combined_df['P_count'] = combined_df['ground_truth'].str.count('P')
    combined_df['Word_count'] = combined_df['ground_truth'].str.split().str.len()

    # rename columns to match the current structure
    combined_df.to_csv(target_path, encoding='utf-8_sig', index=False)
    return combined_df


if __name__ == '__main__':
    folder = r"~\datasets\chunking_paper_ALP2025"
    gt_path = Path(folder) / "ALP2025_GT.csv"
    hyps_path = Path(folder) / "ALP2025_data.csv"
    combined_path = Path(folder) / "ALP2025_combined.csv"

    round1_ids = [1, 2, 3, 4, 5, 6, 7, 8, 16, 19, 29, 39, 49, 59, 69]

    combine_gt_with_hyps(gt_path=gt_path, hyps_path=hyps_path, target_path=combined_path)

    analysis_df = process_all(path=combined_path, id_col='source_id', cite_col='source',
                              hyp_col='hyp', ref_col='ground_truth',
                              metadata_cols=['name'])

    analysis_df = add_specialized_cross_f1(analysis_df, precision_tag='BPM', recall_tag='B', new_col='f1_pBPM_rB')

    analysis_path = Path(folder) / "ALP2025_analysis.csv"
    analysis_df.to_csv(analysis_path, encoding='utf-8_sig', index=False)
