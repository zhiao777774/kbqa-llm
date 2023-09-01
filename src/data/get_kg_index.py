import json
import pandas as pd

if __name__ == '__main__':
    triplets_df = pd.read_json('src/data/KQAPro/kg/KQAPro_kb_triplets.json')
    triplets_df.columns = ['v1', 'relation', 'v2']
    triplets_df['v2'] = triplets_df['v2'].values.astype(str)

    relation_index = triplets_df['relation'].unique().tolist()
    all_entities = pd.concat([triplets_df['v1'], triplets_df['v2']
                              ])  # all entities contains attribute values
    entity_index = all_entities.unique().tolist()
    # # 特殊處理，由於 platform 同時出現在 relation 和 entity 中，導致產生id衝突
    # entity_index[entity_index.index('platform')] = 'Platform'

    all_index = relation_index + entity_index

    print(len(relation_index))
    print(len(entity_index))
    print(len(all_index))

    with open('src/data/KQAPro/kg/KQAPro_kb_entities.json', 'w') as f:
        json.dump(entity_index, f)

    with open('src/data/KQAPro/kg/KQAPro_kb_relations.json', 'w') as f:
        json.dump(relation_index, f)

    with open('src/data/KQAPro/kg/KQAPro_kb_all.json', 'w') as f:
        json.dump(all_index, f)