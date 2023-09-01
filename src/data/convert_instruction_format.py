import json
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm


def extract_sparql(data, instruction):
    results = []

    print('Extracting SPARQL...')
    for item in tqdm(data.iterrows()):
        question = item['question']
        sparql = item['sparql']

        results.append({
            'instruction': instruction,
            'input': question,
            'output': sparql
        })

    return results


def extract_qa(data, instruction, lang):
    """Extract QA from data.

    Args:
        data (DataFrame): _description_
        instruction (str): _description_
        lang (str): _description_

    Returns:
        _type_: _description_
    """
    results = []

    print('Extracting QA...')
    for item in tqdm(data.iterrows()):
        question = item['question']
        answer = item['answer']
        choices = item['choices']

        inp = f'[問題]\r\n{question}\r\n[選項]\r\n{choices}' if lang == 'tw' else \
        f'[Question]\r\n{question}\r\n[Choices]\r\n{choices}'

        results.append({
            'instruction': instruction,
            'input': inp,
            'output': answer
        })

    return results


def main(data, output_file_path, type, lang):
    results = None

    if type == 'qa':
        instruction = {
            'tw':
            '請根據以下給定的問題及選項，從中選擇最佳答案。',
            'en':
            'Please choose the best answer from the given question and choices below.'
        }[lang]
        results = extract_qa(data, instruction, lang)
    elif type == 'sparql':
        instruction = {
            'tw':
            '請根據以下給定的問題，生成對應的SPARQL查詢語句',
            'en':
            'Please generate the corresponding SPARQL query statement based on the given question below.'
        }[lang]
        results = extract_sparql(data, instruction)
    else:
        raise ValueError('type must be one of "qa" or "sparql"')

    with open(output_file_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file_path',
                        type=str,
                        default='src/data/KQAPro/train.json')
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('tpye',
                        type=str,
                        default='qa',
                        choices=['qa', 'sparql'])
    parser.add_argument('lang', type=str, default='tw', choices=['tw', 'en'])
    args = parser.parse_args()

    data = pd.read_json(args.input_file_path)

    main(data, args.output_file_path, args.type, args.lang)
