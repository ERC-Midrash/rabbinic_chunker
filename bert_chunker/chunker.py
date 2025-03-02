import re
from transformers import AutoTokenizer, BertForMaskedLM
import torch


CHUNK_MARK = ' // '


class Chunker(object):

    def __init__(self, model_path, max_options=5, bert_input_len=100, max_chunk_len=50, separators='.'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForMaskedLM.from_pretrained(model_path, device_map='cuda').half()
        self.model = BertForMaskedLM.from_pretrained(model_path, device_map='cuda', torch_dtype=torch.float16)
        self.model.eval()
        self.max_options = max_options
        self.bert_input_len = bert_input_len
        self.max_chunk_len = max_chunk_len
        self.separators = separators

    def guess_period_locations(self, sentence: str, num_options: int = None, sufficient_rank: int = None, prefix_string=None, verbose=False):
        num_options = num_options or 1
        sufficient_rank = sufficient_rank or 1
        if prefix_string:
            words = ('[CLS] ' + prefix_string + '. ' + sentence + ' [SEP]').split()
        else:
            words = ('[CLS] ' + sentence + ' [SEP]').split()
        if verbose:
            print(words)
        MASK = '[MASK]'
        guesses = []
        ranked_locations = []
        current_loc_guess = len(words)  # assume all the way at the end
        current_ranking = num_options + 1  # no evidence yet

        options_dict = dict()

        batch_size = 16  # Adjust the batch size as needed

        for i in range(2, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            batch_masked = []
            for j in range(len(batch_words)):
                masked = ' '.join(words[:i + j] + [MASK] + words[i + j:])
                batch_masked.append(masked)

            if verbose:
                print(batch_masked)

            input_ids = self.tokenizer.batch_encode_plus(batch_masked, return_tensors='pt', padding=True)
            input_ids = input_ids['input_ids'].cuda()

            predictions = self.model(input_ids)[0]

            mask_indices = torch.where(input_ids == self.tokenizer.mask_token_id)
            mask_logits = predictions[mask_indices]

            top_options = torch.topk(mask_logits, num_options, dim=1)
            top_opt_ids = top_options.indices.tolist()

            for idx, opt_ids in enumerate(top_opt_ids):
                options_dict[i+idx] = self.tokenizer.convert_ids_to_tokens(opt_ids)

            # options_dict[i] = top_options

        for i, top_options in options_dict.items():
            for rank, opt in enumerate(top_options):
                if len(opt) == 1 and self.separators.find(opt) != -1:  # the option equals one of the separators
                    punct_option = ' '.join(words[:i]) + f'{CHUNK_MARK} ' + ' '.join(words[i:])
                    guesses.append(punct_option)
                    ranked_locations.append(rank)
                    if current_ranking > sufficient_rank and current_ranking > rank:
                        current_loc_guess = i
                        current_ranking = rank
                    break
            else:
                ranked_locations.append(None)
        main_guess = ' '.join(words[:current_loc_guess]) + f'{CHUNK_MARK} ' + ' '.join(words[current_loc_guess:])
        return current_loc_guess-1, ranked_locations, main_guess, guesses  # current_guess counts also the [CLS]

    def chunk_sentence(self, sentence):
        chunks = []
        while sentence not in ('', None):
            # print(sentence)
            best_loc, _, _, _ = self.guess_period_locations(sentence=sentence)
            next_chunk = ' '.join(sentence.split()[:best_loc]) + CHUNK_MARK
            chunks.append(next_chunk)
            sentence = ' '.join(sentence.split()[best_loc:]).strip()
        return chunks

    def chunk_section(self, section: str, max_words=None, betatest=False) -> list:
        """
        Given a section, this method will return a breakdown of the section into a list of chunks.
        :return:
        """
        max_words = max_words or self.max_chunk_len
        chunks = []
        section_words = section.split()
        part_words = []
        while len(section_words) > 0:
            num_options = 1  # initialize
            num_next_words = self.bert_input_len-len(part_words)
            part_words = part_words + section_words[:num_next_words]
            section_words = section_words[num_next_words:]
            while True:
                best_loc, loc_list, _, _ = self.guess_period_locations(sentence=' '.join(part_words),
                                                                       num_options=num_options, sufficient_rank=num_options)
                if best_loc <= max_words or num_options >= self.max_options:
                    break
                num_options += 1

            next_chunk = ' '.join(part_words[:best_loc]) + CHUNK_MARK
            # print(next_chunk)
            part_words = part_words[best_loc:]
            chunks.append(next_chunk)

        if betatest:  # for trying out new stuff
            num_options = 3  ## a bit more proactive at the end
            _, loc_list, _, _ = self.guess_period_locations(sentence=' '.join(part_words),
                                                            num_options=num_options, sufficient_rank=num_options)
            # print(loc_list)
            loc_list = [i+1 for i, value in enumerate(loc_list) if value is not None]
            # print(loc_list)
            loc_list = [0] + loc_list

        else:
            loc_list = [i-best_loc+1 for i, value in enumerate(loc_list) if value is not None]

        if len(loc_list) > 1:
            loc_list.append(len(part_words))
            last_chunks = [' '.join(part_words[loc_list[i]:loc_list[i+1]]) + CHUNK_MARK for i in range(len(loc_list)-1)]
            chunks += last_chunks

        chunks = [re.sub(' +', ' ', chunk) for chunk in chunks]
        return chunks


def clean_text(text, clean_pattern=',|\.|:|˙'):
    text = re.sub(' - ', ' ', text)
    return re.sub(clean_pattern, '', text)
