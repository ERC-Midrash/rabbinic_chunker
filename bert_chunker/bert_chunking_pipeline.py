from bert_chunker import chunker
import pandas as pd

BEREL = 'dicta-il/BEREL'


MODELS = {
    'berel': BEREL,
}

SEPARATORS = {
    'period': '.',
    'period_and_colon': '.:'
}


def process_text(text_chunker: chunker.Chunker, text: str, betatest=False):
    chunks = text_chunker.chunk_section(section=text, betatest=betatest)
    return ' '.join(chunks)


def grid_testing(df, models, separators, num_opts, betatest=False):
    full_df = pd.DataFrame(columns=['category', 'source', 'model', 'separators', 'max_options', 'text',
                                    'cleaned_text', 'chunked_text'])

    if 'cleaned_text' not in df.columns:
        df['cleaned_text'] = df['text'].apply(chunker.clean_text)

    for num_opt in num_opts:
        for seps in separators:
            for model in models:
                print(f"Running: {model} -- {seps} -- {num_opt}")
                temp_df = df.copy()
                temp_df['model'] = model
                temp_df['separators'] = seps
                temp_df['max_options'] = num_opt
                if model in ['snipbert', 'manuscriptbert']:
                    text_chunker = chunker.Chunker(model_path=MODELS[model], separators=SEPARATORS[seps], max_options=num_opt,
                                           bert_input_len=32, max_chunk_len=25)
                else:
                    text_chunker = chunker.Chunker(model_path=MODELS[model], separators=SEPARATORS[seps], max_options=num_opt)

                temp_df['chunked_text'] = temp_df.apply(lambda x: process_text(text_chunker, x['cleaned_text'], betatest), axis=1)
                full_df = pd.concat([full_df, temp_df])
                current_path = r"C:\Users\bensh\workspace\datasets\paper_out__checkpoint.csv"
                full_df.to_csv(current_path, encoding='utf-8_sig')
    enrich_chunked_table(full_df)
    return full_df


def enrich_chunked_table(df):
    df['# chunks'] = df['chunked_text'].apply(lambda x: len(x.split(chunker.CHUNK_MARK)))


if __name__ == '__main__':

    path = r"input.csv"  # input texts path here
    outpath = r"output.csv"  # output texts path here

    df = pd.read_csv(path)

    ## Full
    models = list(MODELS)
    separators = list(SEPARATORS)
    max_opts = [5, 15]


    print(df)
    chunked_df = grid_testing(df=df, models=models, separators=separators, num_opts=max_opts)
    chunked_df.to_csv(outpath, encoding='utf-8_sig')
