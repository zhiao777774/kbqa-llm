import json
from argparse import ArgumentParser


def convert_inscturction_to_chat_format(input_json):
    output_json = []
    for item in input_json:
        conversation = [{
            'from': 'human',
            'value': item['input']
        }, {
            'from': 'gpt',
            'value': item['output']
        }]
        new_item = {
            'id': 'kqapro-tw_en-align',
            'conversations': conversation,
            'instruction': item['instruction']
        }
        output_json.append(new_item)

    return output_json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_json_data = json.load(f)

    output_json_data = convert_inscturction_to_chat_format(input_json_data)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_json_data, f, ensure_ascii=False, indent=4)

    print('JSON conversion completed. Output written to', output_file_path)