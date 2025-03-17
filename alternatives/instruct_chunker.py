from tqdm import tqdm
import json
import re
import pandas as pd
import sys
from string import Template

from anthropic import Anthropic
from openai import OpenAI
import creds
from concurrent.futures import ThreadPoolExecutor, as_completed


CLAUDE = 'claude'
GPT = 'gpt'

CHUNK_MARK = ' // '

PROMPT = ("please take the following unpunctuated text, and punctuate it. "
          "Punctuation includes periods, commas, question marks, semicolons and colons. "
          "Other than punctuating, keep the text exactly as it is. If a word is clipped at the end, like \"אמ'\", leave it like that. "
          "Return the translation punctuated text between <punctuation> tags:\n\n$text\n\nPunctuated text:")


def claude_sonnet(text, print_output=True, model=None):
    ANTHROPIC_MODEL = 'claude-3-5-sonnet-20240620'
    # ANTHROPIC_MODEL = 'claude-3-5-sonnet-20241022'
    model = model or ANTHROPIC_MODEL
    client = Anthropic(api_key=creds.CLAUDE_KEY)

    prompt = Template(PROMPT).substitute({'text': text})

    messages = [{'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': '<punctuation>'}]
    response = client.messages.create(
        model=model,
        messages=messages,
        system='You are an expert in hebrew and rabbinic language, syntax and punctuation.',
        temperature=0,
        max_tokens=4096
    )
    punctuated = response.content[0].text
    if not punctuated.startswith('<punctuation>'):
        punctuated = f'<punctuation>{punctuated}'
    punctuated = re.findall(pattern='<punctuation>(.+?)</punctuation>', string=punctuated, flags=re.DOTALL)[0].strip()

    if print_output:
        translation_json = {'in': text, 'out': punctuated, 'chunk_id': 1}
        json.dump(obj=translation_json, fp=sys.stdout, indent=4, ensure_ascii=False)
        sys.stdout.write('\n')

    return punctuated


def gpt4o(text, print_output=True, model=None):
    OPENAI_MODEL = 'gpt-4o'
    model = model or OPENAI_MODEL
    prompt = Template(PROMPT).substitute({'text': text})

    messages = [
        {"role": "system", "content": "You are an expert in hebrew and rabbinic language, syntax and punctuation."},
        {"role": "user", "content": prompt}
    ]

    openai_client = OpenAI(api_key=creds.OPENAI_KEY)

    gpt4_response = openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        max_tokens=4096
    )
    punctuated = gpt4_response.choices[0].message.content.strip()
    punctuated = re.findall(pattern='<punctuation>(.+?)</punctuation>', string=punctuated, flags=re.DOTALL)[0]

    if print_output:
        translation_json = {'in': text, 'out': punctuated, 'chunk_id': 1}
        json.dump(obj=translation_json, fp=sys.stdout, indent=4, ensure_ascii=False)
        sys.stdout.write('\n')

    return punctuated


def llama31(text, print_output=True, model=None):
    DEFAULT_MODEL = 'meta-llama/llama-3.1-405b-instruct'
    model = model or DEFAULT_MODEL
    prompt = Template(PROMPT).substitute({'text': text})

    messages = [
        {"role": "system", "content": "You are an expert in hebrew and rabbinic language, syntax and punctuation."},
        {"role": "user", "content": prompt}
    ]

    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=creds.OPENROUTER_KEY)

    response = openrouter_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        max_tokens=4096
    )
    punctuated = response.choices[0].message.content.strip()
    punctuated = re.findall(pattern='<punctuation>(.+?)</punctuation>', string=punctuated, flags=re.DOTALL)[0]

    if print_output:
        translation_json = {'in': text, 'out': punctuated, 'chunk_id': 1}
        json.dump(obj=translation_json, fp=sys.stdout, indent=4, ensure_ascii=False)
        sys.stdout.write('\n')

    return punctuated


def chunk_by_punctuation(punct_text, chunk_punct=('\.', ':', ';', '\?'), ignore_punct=(',')):
    chunk_re = '|'.join(chunk_punct)
    ignore_re = '|'.join(ignore_punct)

    # remove start/end quotation marks
    text = re.sub(pattern=r'(\W)"(\w)', repl=r'\1\2', string=punct_text, flags=re.DOTALL)
    text = re.sub(pattern=r'(\w)"(\W)', repl=r'\1\2', string=text, flags=re.DOTALL)
    text = re.sub(pattern=r'(\W)"(\W)', repl=r'\1\2', string=text, flags=re.DOTALL)

    text = re.sub(pattern=ignore_re, repl='', string=text, flags=re.DOTALL)
    text = re.sub(pattern=chunk_re, repl=CHUNK_MARK, string=text, flags=re.DOTALL)
    text = re.sub(pattern='  ', repl=' ', string=text, flags=re.DOTALL)

    return text


def chunk_text(text, text_id, service):
    if service == 'gpt':
        punct_text = gpt4o(text, print_output=False)
    elif service == 'claude':
        punct_text = claude_sonnet(text, print_output=False)
    else:
        return None, None
    chunked_text = chunk_by_punctuation(punct_text=punct_text)
    return chunked_text, text_id


def chunk_texts(text_entries, service, workers=3, tempoutpath=None):
    tasks = []
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for text_entry in text_entries:
            tasks.append(
                executor.submit(chunk_text, text_entry['text'], text_entry['text_id'], service)
            )
        for task in tqdm(as_completed(tasks)):
            try:
                chunked_text, text_id = task.result()
                results.append({'text_id': text_id, 'text': chunked_text})
                if tempoutpath is not None and len(results) % 1000 == 0:
                    df = pd.DataFrame(results)
                    df = df.sort_values(by='text_id')
                    df.to_csv(tempoutpath, encoding='utf-8_sig', index=False)
                    print(f'saved checkpoint: {len(results)} results')
            except:
                pass
    return results


if __name__ == '__main__':
    text = "ויאמרו הבה נבנה לנו עיר ומגדל וראשו בשמים ונעשה לנו שם ר' חייא פתח והרשעים כים נגרש וגו' וכי אית ים נגרש אין דכד ימא נפקא מתקוניה ואזיל בלא חבלא כדין נגרש ואתתרך מאתריה כמאן דרוי חמרא ולא יתיב על בורייה וסלקא ונחתא מ\"ט בגין כי השקט לא יוכל ויגרשו מימיו רפש וטיט דמפקו מימוי כל ההוא טינא דימא וכל טנופא לשפוותיה כגוונא דא אינון רשעים דנפקא מארחא דתקנא ואזלי כרוי חמרא בלא תקונא דנפקי מאורח מישר לאורח עקים מאי טעמא בגין כי השקט לא יוכל דהא עקימו דארחייהו גרים לון למהך בלא תקונא ובלא שכיכו ולא עוד אלא דכל רוגזא דידהו בשעתא דאמרי מלה מפומייהו ההוא מלה רפש וטיט כלהו מפקי טנופא וגיעולא מפומייהו לבר עד דמסתאבי ומסאבי לון ת\"ח ויאמרו הבה נבנה לנו עיר ומגדל וראשו בשמים לית הבה אלא הזמנה בעלמא נבנה לנו עיר ומגדל וראשו בשמים כלהו בעיטא בישא אתו לסרבא ביה בקודשא בריך הוא בשטותא אתו בטפשו דלבא א\"ר אבא שטותא נסיבו בלבייהו אבל בחכמה דרשיעו אתו בגין לנפקא מרשו עלאה לרשו אחרא ולאחלפא יקריה ליקרא נוכראה ובכלא אית רזא דחכמתא עלאה הבה נבנה לנו עיר ומגדל ת\"ח כד מטו להאי בקעה דאיהו רשו נוכראה ואתגלי להו אתר דשלטנותא דא תקיע בגו נוני ימא אמרו הא אתר למיתב ולאתקפה לבא לאתהנאה ביה תתאי מיד הבה נבנה לנו עיר נתקין באתר דא עיר ומגדל ונעשה לנו שם אתר דא יהא לן לדחלא ולא אחרא ונבנה לאתר דא עיר ומגדל למה לן לסלקא לעילא דלא ניכול לאתהנאה מנה הא הכא אתר מתקנא ונעשה לנו שם דחלא למפלח תמן פן נפוץ לדרגין אחרנין ונתבדר לסטרי עלמא"
    # print(text)
    # claude_sonnet(text, print_output=True)
    punct_text = gpt4o(text, print_output=False)
    # punct_text = "ויאמרו: \"הבה נבנה לנו עיר ומגדל וראשו בשמים ונעשה לנו שם.\" ר' חייא פתח: \"והרשעים כים נגרש וגו'.\" וכי אית ים נגרש? אין, דכד ימא נפקא מתקוניה ואזיל בלא חבלא, כדין נגרש ואתתרך מאתריה כמאן דרוי חמרא ולא יתיב על בורייה, וסלקא ונחתא. מ\"ט? בגין \"כי השקט לא יוכל ויגרשו מימיו רפש וטיט,\" דמפקו מימוי כל ההוא טינא דימא וכל טנופא לשפוותיה. כגוונא דא אינון רשעים דנפקא מארחא דתקנא ואזלי כרוי חמרא בלא תקונא, דנפקי מאורח מישר לאורח עקים. מאי טעמא? בגין \"כי השקט לא יוכל,\" דהא עקימו דארחייהו גרים לון למהך בלא תקונא ובלא שכיכו. ולא עוד, אלא דכל רוגזא דידהו בשעתא דאמרי מלה מפומייהו, ההוא מלה \"רפש וטיט,\" כלהו מפקי טנופא וגיעולא מפומייהו לבר עד דמסתאבי ומסאבי לון. ת\"ח: \"ויאמרו הבה נבנה לנו עיר ומגדל וראשו בשמים.\" לית \"הבה\" אלא הזמנה בעלמא. \"נבנה לנו עיר ומגדל וראשו בשמים,\" כלהו בעיטא בישא אתו לסרבא ביה בקודשא בריך הוא, בשטותא אתו בטפשו דלבא. א\"ר אבא: שטותא נסיבו בלבייהו, אבל בחכמה דרשיעו אתו בגין לנפקא מרשו עלאה לרשו אחרא, ולאחלפא יקריה ליקרא נוכראה. ובכלא אית רזא דחכמתא עלאה. \"הבה נבנה לנו עיר ומגדל.\" ת\"ח: כד מטו להאי בקעה דאיהו רשו נוכראה ואתגלי להו אתר דשלטנותא דא תקיע בגו נוני ימא, אמרו: \"הא אתר למיתב ולאתקפה לבא לאתהנאה ביה תתאי.\" מיד: \"הבה נבנה לנו עיר, נתקין באתר דא עיר ומגדל ונעשה לנו שם.\" אתר דא יהא לן לדחלא ולא אחרא, ונבנה לאתר דא עיר ומגדל. למה לן לסלקא לעילא דלא ניכול לאתהנאה מנה? הא הכא אתר מתקנא. \"ונעשה לנו שם,\" דחלא למפלח תמן, \"פן נפוץ\" לדרגין אחרנין ונתבדר לסטרי עלמא."
    print(punct_text)
    chunked_text = chunk_by_punctuation(punct_text=punct_text)
    print(chunked_text)
