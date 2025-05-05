from string import Template
from ...utils.llm_utils import convert_format_to_template

disambiguation = """Given an entity as a set of facts, your task is to write a simple, short and concise description/definition of the entity.""" 

prompt_template_one_shot = """Given an entity as a set of facts, write a simple, short and concise description/definition of the entity.

Entity: {named_entity}

Passage: ```
{passage}
```
"""

one_shot_entity = "Apple"
one_shot_paragraph = "Apple Inc. is an American multinational corporation and technology company headquartered in Cupertino, California, in Silicon Valley. It is best known for its consumer electronics, software, and services. Founded in 1976 as Apple Computer Company by Steve Jobs, Steve Wozniak and Ronald Wayne, the company was incorporated by Jobs and Wozniak as Apple Computer, Inc. the following year. It was renamed Apple Inc. in 2007 as the company had expanded its focus from computers to consumer electronics. Apple is the largest technology company by revenue, with US$$391.04 billion in the 2024 fiscal year."

one_shot_output = "a technology company."

# Convert the format-style strings to Template-compatible strings
user_template_str = convert_format_to_template(original_string=prompt_template_one_shot, placeholder_mapping=None, static_values=None)
# print(user_template_str)
# raise Exception("test")
one_shot_template_str = prompt_template_one_shot.format(named_entity=one_shot_entity, passage=one_shot_paragraph)

prompt_template = [
    {"role": "system", "content": disambiguation},
    {"role": "user", "content": one_shot_template_str},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": user_template_str}
]