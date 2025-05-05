disambiguation = """Given an entity as a set of facts, write a short, simple and concise description of the entity.""" 

prompt_template_one_shot = """Entity: {entity}

Facts:
{facts}

A short, simple and concise description:"""

prompt_template = """Entity: ${entity}

Facts:
${facts}

A short, simple and concise description:"""

one_shot_entity = "The University of Southampton"
one_shot_facts = [("The University of Southampton", "a public research university located in Southampton, England."), ("It was founded in 1862.")]

one_shot_output = "a public research university in Southampton, England, founded in 1862."

prompt_template = [
    {"role": "system", "content": disambiguation},
    {"role": "user", "content": prompt_template_one_shot.format(entity=one_shot_entity, facts=one_shot_facts)},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": prompt_template}
]