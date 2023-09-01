import os
import sys
import json
from tqdm import tqdm
from kopl.kopl import KoPLEngine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KOPL_ENGINE = KoPLEngine(json.load(open('src/data/KQAPro/kb.json')))


def get_program_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        args = ''
        for input in inputs:
            args += ' <arg> ' + input
        seq.append(func + args)
    seq = ' <func> '.join(seq)
    return seq


def kopl_query(program_seq, executor):
    chunks = program_seq.split('<func>')
    func_list = []
    inputs_list = []
    for chunk in chunks:
        chunk = chunk.strip()
        res = chunk.split('<arg>')
        res = [_.strip() for _ in res]
        if len(res) > 0:
            func = res[0]
            inputs = []
            if len(res) > 1:
                for x in res[1:]:
                    inputs.append(x)
            else:
                inputs = []
            func_list.append(func)
            inputs_list.append(inputs)

    ans = executor.forward(func_list, inputs_list, ignore_error=True)
    return ans


def convert_kg_triplets():
    triplets = []
    entities = KOPL_ENGINE.kb.entities

    print('Converting KoPL to triplets...')
    for entid, entinfo in tqdm(entities.items()):
        name = entinfo['name']
        relations = entinfo['relations']
        attributes = entinfo['attributes']

        for relinfo in tqdm(relations,
                            desc=f' {name}\'s relations:',
                            leave=False):
            relation = relinfo['relation']
            direction = relinfo['direction']
            object_id = relinfo['object']

            if direction == 'forward':
                triplet = (entid, relation, object_id)
            elif direction == 'backward':
                triplet = (object_id, relation, entid)

            if triplet not in triplets:
                triplets.append(triplet)

        for attrinfo in tqdm(attributes,
                             desc=f' {name}\'s attributes:',
                             leave=False):
            key = attrinfo['key']
            value = attrinfo['value']

            if value.type == 'date':
                triplets.append((entid, key, str(value.value)))
            else:
                triplets.append((entid, key, value.value))

    print(f'Found {len(triplets)} triplets.')
    return triplets


if __name__ == '__main__':
    import random as rd

    train_set = json.load(open('src/data/KQAPro/train.json'))

    # seq = get_program_seq([{
    #     "function": "Find",
    #     "inputs": ["financial crisis"],
    #     "dependencies": [-1, -1]
    # }, {
    #     "function": "QueryAttr",
    #     "inputs": ["point in time"],
    #     "dependencies": [0, -1]
    # }])

    # print(kopl_query(seq, KOPL_ENGINE))

    print(len(KOPL_ENGINE.kb.entities))

    while True:
        n = int(input('Enter a number: (press -1 to exit))'))
        if n == -1:
            break

        # n = rd.randint(0, len(train_set))
        data = train_set[n]
        program = data['program']
        program_seq = get_program_seq(program)

        print(f'Question: {data["question"]}')
        print(f'Program: {program_seq}')
        print(f'Answer: {data["answer"]}')
        print(f'Predicted answer: {kopl_query(program_seq, KOPL_ENGINE)}')
