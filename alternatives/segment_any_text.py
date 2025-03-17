from wtpsplit import SaT
import pandas as pd


def get_dataset_from_csv(path, num_samples=None):
    df = pd.read_csv(path, encoding='utf-8')
    df_dict = df.to_dict('records')
    if num_samples is not None:
        df_dict = df_dict[:num_samples]
    return df_dict


datapath = r"~\datasets\chunking_paper_input.csv"
text_entries = get_dataset_from_csv(datapath)
text_entries = sorted(text_entries, key=lambda x: int(x['text_id']))

texts = [te['text'] for te in text_entries]

sat = SaT("sat-12l-sm")

split_iterator = sat.split(texts)
results = []

# add // as delims
for i, splitted in enumerate(split_iterator):
    print(i)
    chunked = r' // '.join(splitted)
    input_dict = text_entries[i]
    input_dict['chunked_text'] = chunked
    results.append(input_dict)

outfile = r'~\datasets\chunking_paper__SegmentAnyText.csv'

df = pd.DataFrame(results)
df.rename(columns={'text': 'chunked_text'}, inplace=True)
df = df.sort_values(by='text_id')
df.to_csv(outfile, encoding='utf-8_sig', index=False)

