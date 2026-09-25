import ast
import json


NER_SYSTEM = "Your task is to extract named entities from the given paragraph. Respond with a JSON list of entities."
NER_EXAMPLE = (
    "Radio City\nRadio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music "
    "portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."
)
NER_ANSWER = '{"named_entities":\n    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]\n}'
QUERY_SYSTEM = "You're a very effective entity extraction system."
QUERY_EXAMPLE = (
    "Please extract all named entities that are important for solving the questions below.\n"
    "Place the named entities in json format.\n\n"
    "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
)
QUERY_ANSWER = '\n{"named_entities": ["First for Women", "Arthur\'s Magazine"]}\n'
TRIPLE_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from "
    "the given passages and named entity lists. Respond with a JSON list of triples, "
    "with each triple representing a relationship in the RDF graph. \n\n"
    "Pay attention to the following requirements:\n"
    "- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.\n"
    "- Clearly resolve pronouns to their specific names to maintain clarity.\n"
)
TRIPLE_FRAME = "Convert the paragraph into a JSON dict, it has a named entity list and a triple list.\nParagraph:\n```\n{passage}\n```\n\n{entities}\n"
TRIPLE_ANSWER = json.dumps({"triples": [
    ["Radio City", "located in", "India"],
    ["Radio City", "is", "private FM radio station"],
    ["Radio City", "started on", "3 July 2001"],
    ["Radio City", "plays songs in", "Hindi"],
    ["Radio City", "plays songs in", "English"],
    ["Radio City", "forayed into", "New Media"],
    ["Radio City", "launched", "PlanetRadiocity.com"],
    ["PlanetRadiocity.com", "launched in", "May 2008"],
    ["PlanetRadiocity.com", "is", "music portal"],
    ["PlanetRadiocity.com", "offers", "news"],
    ["PlanetRadiocity.com", "offers", "videos"],
    ["PlanetRadiocity.com", "offers", "songs"]
]}, indent=4)


def messages(system, example, answer, text):
    return [{"role": "system", "content": system}, {"role": "user", "content": example},
            {"role": "assistant", "content": answer}, {"role": "user", "content": text}]


def ner_messages(text, query=False):
    if query:
        return messages(QUERY_SYSTEM, QUERY_EXAMPLE, QUERY_ANSWER, "Question: " + text)
    return messages(NER_SYSTEM, NER_EXAMPLE, NER_ANSWER, text)


def triple_messages(text, entities):
    return messages(TRIPLE_SYSTEM, TRIPLE_FRAME.format(passage=NER_EXAMPLE, entities=NER_ANSWER),
                    TRIPLE_ANSWER, TRIPLE_FRAME.format(passage=text, entities=json.dumps({"named_entities": entities})))


def parse_response(text):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        fragment = text[start:]
        try:
            return decoder.raw_decode(fragment)[0]
        except json.JSONDecodeError:
            pass
        stack, quoted, escaped = [], False, False
        for end, current in enumerate(fragment):
            if escaped:
                escaped = False
            elif current == "\\" and quoted:
                escaped = True
            elif current == '"':
                quoted = not quoted
            elif not quoted:
                if current in "[{":
                    stack.append("]" if current == "[" else "}")
                elif current in "]}":
                    if not stack or stack.pop() != current:
                        break
                    if not stack:
                        try:
                            return ast.literal_eval(fragment[:end + 1])
                        except (ValueError, SyntaxError):
                            break
        cut = fragment.rfind(",")
        if cut < 0:
            continue
        truncated = fragment[:cut]
        stack, quoted, escaped = [], False, False
        for current in truncated:
            if escaped:
                escaped = False
            elif current == "\\" and quoted:
                escaped = True
            elif current == '"':
                quoted = not quoted
            elif not quoted and current in "[{":
                stack.append("]" if current == "[" else "}")
            elif not quoted and current in "]}" and stack:
                stack.pop()
        if not quoted:
            try:
                return json.loads(truncated + "".join(reversed(stack)))
            except json.JSONDecodeError:
                pass
    return {}


def entities_from(text):
    value = parse_response(text)
    values = value.get("named_entities", []) if isinstance(value, dict) else value
    return list(dict.fromkeys(item for item in values if isinstance(item, str))) if isinstance(values, list) else []


def triples_from(text):
    value = parse_response(text)
    values = value.get("triples", []) if isinstance(value, dict) else value
    if not isinstance(values, list):
        return []
    return [list(item) for item in dict.fromkeys(tuple(str(part) for part in item)
            for item in values if isinstance(item, (tuple, list)) and len(item) == 3)]
