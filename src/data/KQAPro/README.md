# KQA Pro version 1.0

KQA Pro is a large-scale dataset of complex question answering over knowledge base. The questions are very diverse and challenging, requiring multiple reasoning capabilities including compositional reasoning, multi-hop reasoning, quantitative comparison, set operations, and etc. Strong supervisions of SPARQL and program are provided for each question.
If you find our dataset is helpful in your work, please cite us by

```
@inproceedings{KQAPro,
  title={{KQA P}ro: A Large Diagnostic Dataset for Complex Question Answering over Knowledge Base},
  author={Cao, Shulin and Shi, Jiaxin and Pan, Liangming and Nie, Lunyiu and Xiang, Yutong and Hou, Lei and Li, Juanzi and He, Bin and Zhang, Hanwang},
  booktitle={ACL'22},
  year={2022}
}
```

## Usage
There are four json files included in our dataset:

- `kb.json`, the target knowledge base used to answer questions, which is a dense subset of [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page).
- `train.json`, the training set, including 94,376 QA pairs with annotations of SPARQL and program for each.
- `val.json`, the validation set, including 11,797 QA pairs with SPARQL and program.
- `test.json`, the test set, including 11,797 questions, with 10 candidate choices for each. You can submit your predictions and your performance will be shown in our leaderboard.

Following is the detailed formats

**kb.json**
```
{
    'concepts': {
        'id': {
            'name': '',
            'subclassOf': ['<concept_id>'],
        }
    },
    # excluding concepts
    'entities': {
        'id': {
            'name': '<entity_name>',
            'instanceOf': ['<concept_id>'],
            'attributes': [ 
                {
                    'key': '<key>',
                    'value': {
                        'type': 'string'/'quantity'/'date'/'year'
                        'value': float/int/str, # float or int for quantity, int for year, 'yyyy/mm/dd' for date
                        'unit':  str,  # for quantity
                    },
                    'qualifiers': {
                        '<qk>': [
                            <qv>, # the format of qualifier value is similar to attribute value
                        ]
                    }
                }
            ]
            'relations': [ 
                {
                    'relation': '<relation>',
                    'direction': 'forward' or 'backward',
                    'object': '<object_id>',
                    'qualifiers': {
                        '<qk>': [
                            <qv>, # the format of qualifier value is similar to attribute value
                        ]
                    }
                }
            ]
        }
    }
}
```

Here is an example kb:
```
example_kb = {
    'concepts': {
        'Q13393265': {
            'name': 'basketball team',
            'subclassOf': ['Q12973014'] 
        },
        'Q12973014': {
            'name': 'sports team',
            'subclassOf': []
        },
        'Q3665646': {
            'name': 'basketball player',
            'subclassOf': ['Q2066131']
        },
        'Q2066131': {
            'name': 'athlete',
            'subclassOf': []
        }
    },
    'entities': {
        'Q36159': {
            'name': 'LeBron James',
            'instanceOf': ['Q3665646'],
            'attributes': [
                {
                    'key': 'height',
                    'value': {
                        'type': 'quantity',
                        'value': 206,
                        'unit': 'centimetre'
                    },
                    'qualifiers': {}
                },
                {
                    'key': 'work period (start)',
                    'value': {
                        'type': 'year',
                        'value': 2003
                    },
                    'qualifiers': {}
                },
                {
                    'key': 'sex or gender',
                    'value': {
                        'type': 'string',
                        'value': 'male'
                    },
                    'qualifiers': {}
                },
                {
                    'key': 'date of birth',
                    'value': {
                        'type': 'date',
                        'value': '1984-12-30'
                    },
                    'qualifiers': {}
                }
            ],
            'relations': [
                {
                    'relation': 'place of birth',
                    'direction': 'forward',
                    'object': 'Q163132',
                    'qualifiers': {}
                }, 
                {
                    'relation': 'drafted by',
                    'direction': 'forward',
                    'object': 'Q162990',
                    'qualifiers': {
                        'point in time': [
                            {
                                'type': 'date',
                                'value': '2003-06-26'
                            }
                        ]
                    }
                },
                {
                    'relation': 'child',
                    'direction': 'forward',
                    'object': 'Q22302425',
                    'qualifiers': {}

                },
                {
                    'relation': 'member of sports team',
                    'direction': 'forward',
                    'object': 'Q162990',
                    'qualifiers': {
                        'position played on team/speciality': [
                            {
                                'type': 'string',
                                'value': 'small forward'
                            }
                        ],
                        'sport number': [
                            {
                                'type': 'quantity',
                                'value': 23,
                                'unit': '1'
                            }
                        ]
                    }
                }
            ]
        },
        'Q163132': {
            'name': 'Akron',
            'instanceOf': [],
            'attributes': [
                {
                    'key': 'population',
                    'value': {
                        'type': 'quantity',
                        'value': 199110,
                        'unit': '1'
                    },
                    'qualifiers': {
                        'point in time': [
                            {
                                'type': 'year',
                                'value': 2010
                            }
                        ]
                    }
                }
            ],
            'relations': []
        },
        'Q162990': {
            'name': 'Cleveland Cavaliers',
            'instanceOf': ['Q13393265'],
            'attributes': [
                {
                    'key': 'inception',
                    'value': {
                            'type': 'year',
                            'value': 1970
                    },
                    'qualifiers': {}
                }
            ],
            'relations': []
        },
        'Q22302425': {
            'name': 'LeBron James Jr.',
            'instanceOf': ['Q3665646'],
            'attributes': [
                {
                    'key': 'height',
                    'value': {
                        'type': 'quantity',
                        'value': 188,
                        'unit': 'centimetre'
                    },
                    'qualifiers': {} 
                },
                {
                    'key': 'sex or gender',
                    'value': {
                        'type': 'string',
                        'value': 'male'
                    },
                    'qualifiers': {}
                },
                {
                    'key': 'date of birth',
                    'value': {
                        'type': 'date',
                        'value': '2004-10-06'
                    },
                    'qualifiers': {}
                }
            ],
            'relations': [
                {
                    'relation': 'father',
                    'direction': 'forward',
                    'object': 'Q36159',
                    'qualifiers': {}
                }
            ]

        }
    }

}

```

**train.json/val.json**
```
[
    {
        'question': str,
        'sparql': str, # executable in our virtuoso engine
        'program': 
        [
            {
                'function': str,  # function name
                'dependencies': [int],  # functional inputs, representing indices of the preceding functions
                'inputs': [str],  # textual inputs
            }
        ],
        'choices': [str],  # 10 answer choices
        'answer': str,  # golden answer
    }
]
```

**test.json**
```
[
    {
        'question': str,
        'choices': [str],  # 10 answer choices
    }
]
```

## How to run SPARQLs and programs
We implement multiple baselines in our [codebase](https://github.com/shijx12/KQAPro_Baselines), which includes a supervised SPARQL parser and program parser.

For the SPARQL parser, we implement a query engine based on [Virtuoso](https://github.com/openlink/virtuoso-opensource.git).
You can install the engine based on our [instructions](https://github.com/shijx12/KQAPro_Baselines/blob/master/SPARQL/README.md), and then feed your predicted SPARQL to get the answer.

For the program parser, we follow the KoPL engine in [KoPL](https://github.com/THU-KEG/KoPL), which receives a predicted program and returns the answer.
Detailed introductions of our functions can be found in our [paper](https://arxiv.org/abs/2007.03875).

## How to submit results of test set
You need to predict answers for all questions of test set and write them in a text file **in order**, one per line.
Here is an example:
```
Tron: Legacy
Palm Beach County
1937-03-01
The Queen
...
```

Then you need to send the prediction file to us by email <caosl19@mails.tsinghua.edu.cn>, we will reply to you with the performance as soon as possible.
To appear in the learderboard, you need to also provide following information:

- model name
- affiliation
- open-ended or multiple-choice
- whether use the supervision of SPARQL in your model or not
- whether use the supervision of program in your model or not
- single model or ensemble model
- (optional) paper link
- (optional) code link


## Contact
If you have any questions, feel free to contact <caosl19@mails.tsinghua.edu.cn>.
