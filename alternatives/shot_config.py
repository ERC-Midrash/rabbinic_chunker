from pathlib import Path
from string import Template
from dataclasses import dataclass

SHOTS_TEST_PATH = Path(r'shots/shots_test.json')


@dataclass
class ChunkConfig:
    name: str
    shots: dict
    raw_prefix: str
    chunked_prefix: str
    shot_num: int = 0


def get_prompt_template(chunk_conf: ChunkConfig):
    prompt = ''

    for idx, shot in enumerate(chunk_conf.shots):
        prompt += f'{chunk_conf.raw_prefix}: {shot["raw"]}\n'
        prompt += f'{chunk_conf.chunked_prefix}: {shot["chunked"]}\n\n'
        if chunk_conf.shot_num > 0 and chunk_conf.shot_num == idx + 1:
            break
    prompt += f'{chunk_conf.raw_prefix}: $raw_text\n'
    prompt += f'{chunk_conf.chunked_prefix}:'

    return Template(prompt)
