import time
import os
import json
import ast
import openai
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chatcompletion_with_backoff(**kwargs) -> openai.ChatCompletion:
    return openai.ChatCompletion.create(**kwargs)


def translate(prompt: str, text: str, model='gpt-3.5-turbo') -> str:
    openai.api_key = os.getenv('OPENAI_API_KEY')

    messages = [{
        'role': 'user',
        'content': prompt.replace('{sentence}', text)
    }]
    completion = chatcompletion_with_backoff(model=model,
                                             max_tokens=1024,
                                             temperature=0,
                                             messages=messages)

    res = completion.choices[0].message.content

    return res


def init_ckpt_file():
    qa = {'tw': [], 'en': []}
    sparql = {'tw': [], 'en': []}

    with open(f'src/output/KQAPro_train_multiplechoice-tw.json', 'r') as f:
        qa['tw'] = json.load(f)

    with open(f'src/output/KQAPro_train_multiplechoice-tw_en_instruction.json',
              'r') as f:
        qa['en'] = json.load(f)

    with open(f'src/output/KQAPro_train_sparql-tw.json', 'r') as f:
        sparql['tw'] = json.load(f)

    with open(f'src/output/KQAPro_train_sparql-tw_en_instruction.json',
              'r') as f:
        sparql['en'] = json.load(f)

    return qa, sparql


if __name__ == '__main__':
    with open('src/prompts/translate.prompt', 'r') as f:
        prompt = f.read()

    data = pd.read_json('src/data/KQAPro/train.json')
    data = data.iloc[1000:]
    qa_results, sparql_results = init_ckpt_file()

    model = 'gpt-4'
    try:
        for i, item in data.iterrows():
            question = item['question']
            choices = item['choices']
            answer = item['answer']
            sparql = item['sparql']

            msg = f'{{"問題": \"{question}\", "選項": {choices}, "答案": \"{answer}\"}}'

            print(i, '=' * 30)
            print(msg)
            print('->')

            res = translate(prompt, msg, model)

            if not res.startswith('{'):
                question, choices_and_answer = res.split('選項:')
                question = question.split('問題:')[1]
                choices, answer = choices_and_answer.split('答案:')

                question = question.strip()
                choices = choices.strip()
                answer = answer.strip()

                res = f'{{"問題": "{question}", "選項": {choices}, "答案": "{answer}"}}'

            pprint(res)
            res = ast.literal_eval(res)

            qa_results['tw'].append({
                'instruction': '請根據以下給定的問題及選項，從中選擇最佳答案。',
                'input': f'[問題]\r\n{res["問題"]}\r\n[選項]\r\n{res["選項"]}',
                'output': res['答案']
            })

            qa_results['en'].append({
                'instruction':
                'Please choose the best answer from the given question and choices below.',
                'input':
                f'[Question]\r\n{res["問題"]}\r\n[Choices]\r\n{res["選項"]}',
                'output': res['答案']
            })

            sparql_results['tw'].append({
                'instruction': '請根據以下給定的問題，生成對應的SPARQL查詢語句',
                'input': res['問題'],
                'output': sparql
            })

            sparql_results['en'].append({
                'instruction':
                'Please generate the corresponding SPARQL query statement based on the given question below.',
                'input': res['問題'],
                'output': sparql
            })

            pprint(res)
            print()
            time.sleep(0.2)

    except ValueError as e:
        print(e)
    finally:
        with open(f'src/output/KQAPro_train_multiplechoice-tw.json', 'w') as f:
            json.dump(qa_results['tw'], f, ensure_ascii=False, indent=4)

        with open(
                f'src/output/KQAPro_train_multiplechoice-tw_en_instruction.json',
                'w') as f:
            json.dump(qa_results['en'], f, ensure_ascii=False, indent=4)

        with open(f'src/output/KQAPro_train_sparql-tw.json', 'w') as f:
            json.dump(sparql_results['tw'], f, ensure_ascii=False, indent=4)

        with open(f'src/output/KQAPro_train_sparql-tw_en_instruction.json',
                  'w') as f:
            json.dump(sparql_results['en'], f, ensure_ascii=False, indent=4)
