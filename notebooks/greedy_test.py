
# %%
import pytest


def pair_greedy(pairs, label_budgets, budget, mapper):
    """
    pairs: (obj_val, id)
    label_budgets: label -> budget
    budget: int
    mapper: id -> label
    """
    label_budgets = {k: v for k, v in label_budgets.items() if v is not None}

    result = set()
    for val, v in pairs:
        if len(result) >= budget:
            break
        label = mapper(v)
        if label not in label_budgets:
            result.add(v)
        else:
            # Add greedily until budget runs out for a particular label
            if label_budgets[label] > 0:
                result.add(v)
                label_budgets[label] -= 1
    return result



# %%

# %%
