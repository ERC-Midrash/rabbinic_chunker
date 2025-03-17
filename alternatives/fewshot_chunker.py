import pandas as pd
from tqdm import tqdm
import json
import re
import requests
from shot_config import get_prompt_template, ChunkConfig, SHOTS_TEST_PATH
from langchain.llms.base import LLM
from langchain.docstore.document import Document
from typing import Optional, List, Mapping, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


LLM_URL = "https://dicta-translation.loadbalancer3.dicta.org.il/whatcanthisbe/completions"


def call_llm(text: str, text_id: int, prompt_template, llm):
    doc = Document(page_content=prompt_template.substitute({'raw_text': text}))
    output = llm._call(doc.page_content)
    return output, text_id


def chunk_texts(text_entries, prompt_template, workers=3, tempoutpath=None):
    tasks = []
    results = []
    llm = CustomLLM(url=LLM_URL, temperature=0.0, max_tokens=2048)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for text_entry in text_entries:
            tasks.append(
                executor.submit(call_llm, text_entry['text'], text_entry['text_id'], prompt_template, llm)
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


class CustomLLM(LLM):
    url: str
    temperature: float
    max_tokens: int

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> str:


        payload = {
            "max_tokens": self.max_tokens,
            "model": "dicta-il/dictalm2.0",
            "prompt": prompt,
            "stop": "\n",
            "stream": True,
            "temperature": self.temperature
        }
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(self.url, json=payload, headers=headers, stream=True)
        chunked = ''
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                data = re.search('data: (.+)', decoded_line).group(1)
                try:
                    json_data = json.loads(data)
                    chunked += json_data['choices'][0]['text']
                except:
                    pass
        chunked = re.sub(' +', ' ', chunked)
        return chunked.strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"url": self.url}

    @property
    def _llm_type(self) -> str:
        return "custom"


if __name__ == '__main__':

    text_entries = [
        {'text': "ויצא משה וידבר אל העם וירד ה' בענן וישארו שני אנשים במחנה יש אומרים בקלפי נשתיירו לפי שאמר לו הקדוש ברוך הוא למשה לבור לו שבעים זקנים אמר משה מה אני עושה הרי הם נופלים",
         'text_id': 80},
        {'text': "מפני מה נענש אברהם אבינו ע\"ה ונשתעבדו בניו כו' אף על פי שעבירת העונש מפורשת על אומרו במה אדע מ\"מ עבירה זו הענישתו להיות נופל בבמה אדע שהוריקן בתורה זרזן בזכות התורה ובשבילה נצחו במלחמה שהוריקן בזהב פתח להם אוצרותיו כדי שילכו עמו בנפש חפצה השטן בגימטריא שס\"ד הוי וימות החמה שס\"ה רמז שכל השנה יש לו רשות לקטרג חוץ מיום אחד והיינו יוה\"כ שתי עינים ושתי אזנים וראש הגוייה והם אברים המסרסרים בעבירה ואינן מסורין ביד האדם ומרוב צדקתו מסרם בידו כדכתיב רגלי חסידיו ישמור כשאדם מתגבר על יצרו סוף שהקב\"ה מוסרו בידו זה יצר הרע כדכתיב מלך זקן וכסיל איש מסכן וחכם זה יצה\"ט כדכתיב טוב ילד מסכן וחכם וזהו יצ\"ט שהוא ילד בשנים כנגד יצה\"ר שהוא פחות ממנו י\"ג שנה וחכמתו בזויה שאין האברים נשמעין לו נאם ה' לאדני אדוני הוא אברהם אבינו שקראו ראשון להקב\"ה אדון כדאיתא בברכות למען אדני למען אברהם שקראו אדון הדרן עלך ארבעה נדרים אין בין המודר הנאה וכו' אלא דריסת הרגל לעבור דרך ארצו או להשאיל כלים שאין עושין בהן אוכל נפש שמותרין במודר הימנו מאכל ואסור במודר הימנו הנאה נפה וכברה רחים ותנור שמתקנין בהן אוכל נפש ואיצטריך לאשמועינן כלים הללו דלא תימא דוקא קדרה ושפוד שהאוכל עומד בתוכו לאכילה אבל הני גורם דגורם מיקרו קמ\"ל דה\"ה רחיים ותנור אבל משאיל לו חלוק וכו' מילתא דפסיקא נקט גמ' מאן תנא דמחמיר במודר הנאה לאסור אף דריסת הרגל אפי' ויתור אסור במודר הנאה מה שדרך אדם לוותר משלו ואינו מקפיד עליו כגון החנווני אחר שנתן המנין או המדה שרגילין ליתן בפונדיון הוא רגיל להוסיף משלו כדי להרגילו לבא אצלו אותו ויתור אסור למודר הנאה כיון שהוא מוותר שלו אבל שוה בשוה מותר דהיינו זבינא דרמי על אפיה וכיון דמחמיר כולי האי הלכך אפי' דריסת הרגל שאין דרך להקפיד עליו אסור במודר",
         'text_id': 7},
        {'text': "גרסינן בפרק איזהו נשך אמר רבה הני בי תלתא דיהבי זוזי לחד למזבן להו מידי וזבן לחד מינייהו זבן לכולהו ולא אמרן אלא דלא צר וחתים איניש לדעתיה כלומר אלא שהיו מעורבין אבל צר וחתים איניש איניש לדעתיה למאן דזבן זבן למאן דלא זבן לא זבן וכ\"כ ר\"ם פ\"ז של קנין וכתב שם הנותן מעות לחבירו לקנות לו קנית קרקע או מטלטלי והניח מעות חבירו אצלו והלך וקנה לעצמו במעותיו מה שעשה עשוי והרי הוא בכלל הרמאין היה יודע שזה המוכר אוהב אותו ומכבדו ומוכר לו ואינו מוכר למשלחו הרי זה מותר לקנות לעצמו והוא שיחזור ויודיעו ואם פחד שמא יבא אחר ויקדמנו לקנות הרי זה מותר לקנות לעצמו ואחר כן מודיעו ע\"כ והיינו מאי דגרסינן פרק האומר דקידושין תניא מה שעשה עשוי אלא שנהג בו מנהג רמאות רבה בר בר חנה יהב זוזי לרב א\"ל זבנה ניהלי להאי ארעא אזל זבנה לנפשיה והתניא מה שעשה עשוי אלא שנהג מנהג רמאות באגא דאלמא הוה פירוש בקעה של אלמים היתה לרב נהגי ביה מנהג כבוד לרבה בר בר חנה לא נהגי ביה מנהג כבוד ואמרינן איבעיא ליה לאודועי סבר אדהכי והכי אתו איניש אחרינא וזבין לה ע\"כ וכתב ר\"מ שאם קנה השליח הבקעה במעותיו של זה שהמקח של המשלח שלא כדעת הגאונים ירושלמי והביאו רי\"ף ז\"ל בפרק איזהו נשך תני [חדא] הנותן מעות לחבירו ליקח בהם פירות למחצית שכר ולא לקח אין לו עליו אלא תערומת ואם ידוע שלקח ומכר ה\"ז מוציא ממנו על כרחו והנותן מעות ליקח בהן פירות למחצית שכר רשאי הלוקח ליקח מכל מין שירצה ולא יקח לא כסות ולא עצים ע\"כ",
         'text_id': 1},
        {'text': "הדרך הב' שהמוכר יכול לחזור בלוקח כגון ש\"מ שמכר נכסיו דלאחר שהבריא אם אותם דמים הם מצויים בעינה ה\"ז יכול לחזור בו ואם לאו אינו יכול לחזור בו כדגר' אבעיא להו ש\"מ שמכר כל נכסיו מהו זמנין אר\"י אמר רב אם עמד חוזר וזמנין אר\"י א\"ר אם עמד אינו חוזר ולא פליגי הא דאיתניהו לזוזי בעיניהו הא דפרעינהו בחובו והם שני הדרכים שפי' שיכול המוכר לחזור בלוקח ואף על פי שהיה הממכר תחלה כתקנו וכמשפטו כ\"ש וכ\"ש אם היה הממכר בתחלה ע\"י אונס דאותו מכר אינו כלום לפי שלא היה כמשפטו ושני הדרכים שהלוקח יכול לחזור במוכר הדרך הא' מי שלקח קרקע מחברו בין באחריות ובין שלא באחריות ורואה שם ערעור אם קודם שהחזיק בקרקע הלוקח הזה יכול לחזור בו מיד ואף על פי שנתן הדמים או שנכתב השטר ואף על פי שהערעור הזה אינו יודע אם יש לו דין בקרקע הזה אם לאו אבל לאחר שהחזיק אינו יכול לחזור במקח עד שירד עם אותו ערער בדין ויטרפו אותו מידו ואח\"כ יחזור על המוכר ודין זה בין שלקחו באחריות בין שלקחו שלא באחריות כדגר' אמר אביי ראובן שמכר שדה לשמעון שלא באחריות ויצאו עליו עסקין עד שלא החזיק בה יכול לחזור בו משהחזיק בה אינו יכול לחזור בו דא\"ל חיתא דקטרי סברת וקבלת ואיכא דאמרי אפילו באחריות נמי דא\"ל אחוי טרפך ואשלם לך ומשום הכי יפה עשה מי שראה להכריז על הקרקע בשעה שיבא ליד הלוקח אם יש מי שיערער עליו או אם יש חילוק עליו כדי שיוכל הלוקח לחזור בו ולא יקרא ואשר לא טוב עשה בתוך עמיו כדגר' ואשר לא טוב עשה בתוך עמיו רב אמר זה הבא בהרשאה ושמואל אמר זה הלוקח שדה שיש עליו עסקין",
            'text_id': 999},
        {
            'text': "תנו רבנן כתב המהלך תחת הצורה והדיוקנאות אסור לקרותו בשבת ודיוקנה עצמה אסור להסתכל בה אף בחול מפיס אדם עם בניו ובני ביתו על השולחן אפילו מנה גדולה כנגד מנה קטנה שמותר להטעימן טעם ריבית אבל עם אחר לא כדרב יהודה אמר שמואל וכו' עד ומשום לוים ופורעין ביום טוב וכו' ומנה גדולה כנגד מנה קטנה לאחר אפילו בחול אסור משום קוביא ומטילין חלשים על הקדשים ביום טוב אבל לא על המנות של חול ביום טוב ואפילו הם קדשים שיש לחוש למריבה כדכתיב ועמך כמריבי כהן",
            'text_id': 987098
        }
    ]
    shots_file = SHOTS_TEST_PATH

    with open(shots_file, encoding='utf-8') as f:
        shots = json.load(f)

    config = ChunkConfig(
        name="test",
        shots=shots,
        raw_prefix='raw_text',
        chunked_prefix='chunked_text'
    )

    results = chunk_texts(text_entries, get_prompt_template(config))
    for result in results:
        print(result)
        print("======================")
