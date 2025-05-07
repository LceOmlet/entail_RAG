from string import Template
from ...utils.llm_utils import convert_format_to_template

disambiguation = """Given an entity as a set of facts, your task is to write a simple, short and concise description/definition of the entity. You should notice that your are writing a concise definition for an entity, instead of summarizing given passages. If you encountered with common entity like numeric numbers, date, time, etc, you should simply return their type "number", "time", "date", etc.""" 

prompt_template_one_shot = """Given an entity as a set of facts, write a simple, short and concise description/definition of the entity. You should notice that your are writing a concise definition for an entity, instead of summarizing given passages. Instead, you should write a general and concise definition with as little as possible details. If you encountered with common entity like numeric numbers, date, time, etc, you should simply return their type "number", "time", "date", etc.

Passage: ```
{passage}
```

Entity: {named_entity}
"""

f0_shot_entity = "Apple"
f0_shot_paragraph = "Apple Inc. is an American multinational corporation and technology company headquartered in Cupertino, California, in Silicon Valley. It is best known for its consumer electronics, software, and services. Founded in 1976 as Apple Computer Company by Steve Jobs, Steve Wozniak and Ronald Wayne, the company was incorporated by Jobs and Wozniak as Apple Computer, Inc. the following year. It was renamed Apple Inc. in 2007 as the company had expanded its focus from computers to consumer electronics. Apple is the largest technology company by revenue, with US$$391.04 billion in the 2024 fiscal year."

f0_shot_output = "a technology company."

f1_shot_entity = "29"
f1_shot_paragraph = "The National Basketball Association (NBA) is a professional basketball league in North America composed of 30 teams (29 in the United States and 1 in Canada). The NBA is one of the major professional sports leagues in the United States and Canada and is considered the premier professional basketball league in the world.[3] The league is headquartered in Midtown Manhattan."

f1_shot_output = "number."

f2_shot_entity = "1946"
f2_shot_paragraph = "The NBA traces its roots to the Basketball Association of America which was founded in 1946 by owners of the major ice hockey arenas in the Northeastern and Midwestern United States and Canada. On November 1, 1946, in Toronto, Ontario, Canada, the Toronto Huskies hosted the New York Knickerbockers at Maple Leaf Gardens, in a game the NBA now refers to as the first game played in NBA history.[11] The first basket was made by Ossie Schectman of the Knickerbockers.[12]"

f2_shot_output = "date."


# Convert the format-style strings to Template-compatible strings
user_template_str = convert_format_to_template(original_string=prompt_template_one_shot, placeholder_mapping=None, static_values=None)
# print(user_template_str)
# raise Exception("test")
f0_shot_template_str = prompt_template_one_shot.format(named_entity=f0_shot_entity, passage=f0_shot_paragraph)
f1_shot_template_str = prompt_template_one_shot.format(named_entity=f1_shot_entity, passage=f1_shot_paragraph)
f2_shot_template_str = prompt_template_one_shot.format(named_entity=f2_shot_entity, passage=f2_shot_paragraph)

prompt_template = [
    {"role": "system", "content": disambiguation},
    {"role": "user", "content": f1_shot_template_str},
    {"role": "assistant", "content": f1_shot_output},
    {"role": "user", "content": f0_shot_template_str},
    {"role": "assistant", "content": f0_shot_output},
    {"role": "user", "content": f2_shot_template_str},
    {"role": "assistant", "content": f2_shot_output},
    {"role": "user", "content": user_template_str}
]