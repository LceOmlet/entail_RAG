entailment_template = "Judge whether the entailment relationship holds between the two clauses.\n$$\\text{{{clause1}}} \\models \\text{{{clause2}}}$$ \nDirectly answer yes/no:" 

prompt_template = [
    {"role": "system", "content": entailment_template},
    {"role": "user", "content": "${prompt_user}"}
]