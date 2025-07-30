import inspect

from .rules1 import FILE_RULES, GROUP_RULES
from .rules2 import NODE_RULES

def explain(args):
    """
    rules = [r for r in STAGE_2_RULES if r.code == args.explain]
    if len(rules) == 0:
        print(f"Rule {args.explain} is not a valid rule")
    else:
        assert len(rules) == 1
        print(f"{rules[0].doc}")
    """
    all_rules = FILE_RULES
    all_rules.update(GROUP_RULES)
    all_rules.update(NODE_RULES)
    all_rules = {key[1:] : val for key,val in all_rules.items()}
    key = args.explain
    if len(key) > 0 and not key[0].isdigit():
        key = key[1:]
    try:
        rule_fn = all_rules[key]
    except KeyError:
        print(f"Rule {args.explain} is not a valid rule")
        exit()
    print(inspect.cleandoc(rule_fn.__doc__))