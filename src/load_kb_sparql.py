import json
from datetime import date
from queue import Queue
from utils.sparql_util import ValueClass


import rdflib
from rdflib import URIRef, BNode, Literal, XSD
from rdflib.plugins.stores import sparqlstore
from itertools import chain
from tqdm import tqdm


def legal(s):
    # convert relation and attribute keys to legal format
    return s.replace(' ', '_')

def esc_escape(s):
    '''
    Why we need this:
    If there is an escape in Literal, such as '\EUR', the query string will be something like '?pv <pred:value> "\\EUR"'.
    However, in virtuoso engine, \\ is connected with E, and \\E forms a bad escape sequence.
    So we must repeat \\, and virtuoso will consider "\\\\EUR" as "\EUR".
    Note this must be applied before esc_quot, as esc_quot will introduce extra escapes.
    '''
    return s.replace('\\', '\\\\')

def esc_quot(s):
    '''
    Why we need this:
    We use "<value>" to represent a literal value in the sparql query.
    If the <value> has a double quotation mark itself, we must escape it to make sure the query is valid for the virtuoso engine.
    '''
    return s.replace('"', '\\"')

class DataForSPARQL(object):
    def __init__(self, kb_path):
        kb = json.load(open(kb_path))
        self.concepts = kb['concepts']
        self.entities = kb['entities']

        # replace adjacent space and tab in name, which may cause errors when building sparql query
        for con_id, con_info in self.concepts.items():
            con_info['name'] = ' '.join(con_info['name'].split())
        for ent_id, ent_info in self.entities.items():
            ent_info['name'] = ' '.join(ent_info['name'].split())

        # get all attribute keys and relations
        self.attribute_keys = set()
        self.relations = set()
        self.key_type = {}
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                self.attribute_keys.add(attr_info['key'])
                self.key_type[attr_info['key']] = attr_info['value']['type']
                for qk in attr_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in attr_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                self.relations.add(rel_info['relation'])

                for qk in rel_info['qualifiers']:
                    self.attribute_keys.add(qk)
                    for qv in rel_info['qualifiers'][qk]:
                        self.key_type[qk] = qv['type']
        self.attribute_keys = list(self.attribute_keys)
        self.relations = list(self.relations)
        # Note: key_type is one of string/quantity/date, but date means the key may have values of type year
        self.key_type = { k:v if v!='year' else 'date' for k,v in self.key_type.items() }

        # parse values into ValueClass object
        for ent_id, ent_info in self.entities.items():
            for attr_info in ent_info['attributes']:
                attr_info['value'] = self._parse_value(attr_info['value'])
                for qk, qvs in attr_info['qualifiers'].items():
                    attr_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]
        for ent_id, ent_info in self.entities.items():
            for rel_info in ent_info['relations']:
                for qk, qvs in rel_info['qualifiers'].items():
                    rel_info['qualifiers'][qk] = [self._parse_value(qv) for qv in qvs]

    def _parse_value(self, value):
        if value['type'] == 'date':
            x = value['value']
            p1, p2 = x.find('/'), x.rfind('/')
            y, m, d = int(x[:p1]), int(x[p1+1:p2]), int(x[p2+1:])
            result = ValueClass('date', date(y, m, d))
        elif value['type'] == 'year':
            result = ValueClass('year', value['value'])
        elif value['type'] == 'string':
            result = ValueClass('string', value['value'])
        elif value['type'] == 'quantity':
            result = ValueClass('quantity', value['value'], value['unit'])
        else:
            raise Exception('unsupport value type')
        return result

    def get_direct_concepts(self, ent_id):
        """
        return the direct concept id of given entity/concept
        """
        if ent_id in self.entities:
            return self.entities[ent_id]['instanceOf']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['instanceOf']
        else:
            raise Exception('unknown id')

    def get_all_concepts(self, ent_id):
        """
        return a concept id list
        """
        ancestors = []
        q = Queue()
        for c in self.get_direct_concepts(ent_id):
            q.put(c)
        while not q.empty():
            con_id = q.get()
            ancestors.append(con_id)
            if 'instanceOf' in self.concepts[con_id]:
                for c in self.concepts[con_id]['instanceOf']:
                    q.put(c)

        return ancestors

    def get_name(self, ent_id):
        if ent_id in self.entities:
            return self.entities[ent_id]['name']
        elif ent_id in self.concepts:
            return self.concepts[ent_id]['name']
        else:
            return None

    def is_concept(self, ent_id):
        return ent_id in self.concepts

    def get_attribute_facts(self, ent_id, key=None, unit=None):
        if key:
            facts = []
            for attr_info in self.entities[ent_id]['attributes']:
                if attr_info['key'] == key:
                    if unit:
                        if attr_info['value'].unit == unit:
                            facts.append(attr_info)
                    else:
                        facts.append(attr_info)
        else:
            facts = self.entities[ent_id]['attributes']
        facts = [(f['key'], f['value'], f['qualifiers']) for f in facts]
        return facts

    def get_relation_facts(self, ent_id):
        facts = self.entities[ent_id]['relations']
        facts = [(f['relation'], f['object'], f['direction'], f['qualifiers']) for f in facts]

        return facts


class SparqlEngine():
    gs1 = None
    PRED_INSTANCE = 'pred:instance_of'
    PRED_NAME = 'pred:name'

    PRED_VALUE = 'pred:value'       # link packed value node to its literal value
    PRED_UNIT = 'pred:unit'         # link packed value node to its unit

    PRED_YEAR = 'pred:year'         # link packed value node to its year value, which is an integer
    PRED_DATE = 'pred:date'         # link packed value node to its date value, which is a date

    PRED_FACT_H = 'pred:fact_h'     # link qualifier node to its head
    PRED_FACT_R = 'pred:fact_r'
    PRED_FACT_T = 'pred:fact_t'

    SPECIAL_PREDICATES = (PRED_INSTANCE, PRED_NAME, PRED_VALUE, PRED_UNIT, PRED_YEAR, PRED_DATE, PRED_FACT_H, PRED_FACT_R, PRED_FACT_T)
    def __init__(self, data, ttl_file=''):
        self.nodes = nodes = {}
        for i in chain(data.concepts, data.entities):
            nodes[i] = URIRef(i)
        for p in chain(data.relations, data.attribute_keys, SparqlEngine.SPECIAL_PREDICATES):
            nodes[p] = URIRef(legal(p))
        
        self.graph = graph = rdflib.Graph()

        for i in chain(data.concepts, data.entities):
            name = data.get_name(i)
            graph.add((nodes[i], nodes[SparqlEngine.PRED_NAME], Literal(name)))

        for ent_id in tqdm(data.entities, desc='Establishing rdf graph'):
            for con_id in data.get_all_concepts(ent_id):
                graph.add((nodes[ent_id], nodes[SparqlEngine.PRED_INSTANCE], nodes[con_id]))
            for (k, v, qualifiers) in data.get_attribute_facts(ent_id):
                h, r = nodes[ent_id], nodes[k]
                t = self._get_value_node(v)
                graph.add((h, r, t))
                fact_node = self._new_fact_node(h, r, t)

                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        h, r = fact_node, nodes[qk]
                        t = self._get_value_node(qv)
                        if len(list(graph[t])) == 0:
                            print(t)
                        graph.add((h, r, t))

            for (pred, obj_id, direction, qualifiers) in data.get_relation_facts(ent_id):
                if direction == 'backward':
                    if data.is_concept(obj_id):
                        h, r, t = nodes[obj_id], nodes[pred], nodes[ent_id]
                    else:
                        continue
                else:
                    h, r, t = nodes[ent_id], nodes[pred], nodes[obj_id]
                graph.add((h, r, t))
                fact_node = self._new_fact_node(h, r, t)
                for qk, qvs in qualifiers.items():
                    for qv in qvs:
                        h, r = fact_node, nodes[qk]
                        t = self._get_value_node(qv)
                        graph.add((h, r, t))

        if ttl_file:
            print('Save graph to {}'.format(ttl_file))
            graph.serialize(ttl_file, format='turtle')


    def _get_value_node(self, v):
        # we use a URIRef node, because we need its reference in query results, which is not supported by BNode
        if v.type == 'string':
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_VALUE], Literal(v.value)))
            return node
        elif v.type == 'quantity': 
            # we use a node to pack value and unit
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_VALUE], Literal(v.value, datatype=XSD.double)))
            self.graph.add((node, self.nodes[SparqlEngine.PRED_UNIT], Literal(v.unit)))
            return node
        elif v.type == 'year':
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_YEAR], Literal(v.value)))
            return node
        elif v.type == 'date':
            # use a node to pack year and date
            node = BNode()
            self.graph.add((node, self.nodes[SparqlEngine.PRED_YEAR], Literal(v.value.year)))
            self.graph.add((node, self.nodes[SparqlEngine.PRED_DATE], Literal(v.value, datatype=XSD.date)))
            return node

    def _new_fact_node(self, h, r, t):
        node = BNode()
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_H], h))
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_R], r))
        self.graph.add((node, self.nodes[SparqlEngine.PRED_FACT_T], t))
        return node

if __name__ == '__main__':
    data = DataForSPARQL('src/data/KQAPro/kb.json')
    engine = SparqlEngine(data, 'src/output/KQAPro_kg.ttl')