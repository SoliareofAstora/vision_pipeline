import copy
import hashlib
import json


EXPERIMENT_AGGREGATOR = []


def select_experiments(keys, value, experiments):
    selection = []
    for i in range(len(experiments)):
        e_value = experiments[i]
        for key in keys:
            e_value = e_value[key]
        if str(e_value) == value:
            selection.append(experiments[i])
    if len(selection) == 0:
        raise RuntimeError('Select experiments returns empty array. Something in the plan is wrong.')
    return selection


def add_param_to_experiments(name, value, experiments, nested_keys=None):
    if nested_keys is None:
        nested_keys = []
    for experiment in experiments:
        for nested_key in nested_keys:
            experiment = experiment[nested_key]
        experiment[name] = copy.deepcopy(value)


def multiply_experiments(name: str, values: list, experiments: list, nested_keys: list):
    new_experiments = []
    for value in values:
        add_param_to_experiments(name, value, experiments, nested_keys)
        new_experiments.extend(copy.deepcopy(experiments))
    return new_experiments


def solve_plan(plan, experiments, group_name='', nested_keys=None):
    if nested_keys is None:
        nested_keys = []
    experiments = copy.deepcopy(experiments)
    group_keys = [x for x in plan.keys() if x.startswith('GROUP:')]
    if len(group_keys) > 0:
        if 'DEFAULT_PARAMETERS' in plan.keys():
            experiments = solve_plan(plan['DEFAULT_PARAMETERS'], experiments)
        for group in group_keys:
            new_group_name = group_name + group.replace('GROUP:', '') + '/'
            add_param_to_experiments('group_name', new_group_name, experiments)
            nested_groups = [x for x in plan[group].keys() if x.startswith('GROUP:')]
            if len(nested_groups) == 0:
                EXPERIMENT_AGGREGATOR.extend(solve_plan(plan[group], experiments, new_group_name))
            else:
                solve_plan(plan[group], experiments, new_group_name)

    else:
        parameter_keys = plan.keys()
        for key in parameter_keys:
            if type(plan[key]) == list and len(plan[key]) > 0:
                if plan[key][0] == 'MULTIPLY':
                    experiments = multiply_experiments(key, plan[key][1], experiments, nested_keys)
                    continue

                if plan[key][0] == 'ADD':
                    check_keys = [plan[key][1]]
                    i = 2
                    while plan[key][i] == 'IN':
                        check_keys = [plan[key][i + 1]] + check_keys
                        i += 2
                    for dkey, value in plan[key][i].items():
                        selection = select_experiments(check_keys, dkey, experiments)
                        add_param_to_experiments(key, value, selection, nested_keys)
                    continue

            elif type(plan[key]) == dict:
                add_param_to_experiments(key, {}, experiments, nested_keys)
                dict_nested_keys = copy.deepcopy(nested_keys)
                dict_nested_keys.append(key)
                experiments = solve_plan(plan[key], experiments, group_name, dict_nested_keys)
                continue

            add_param_to_experiments(key, plan[key], experiments, nested_keys)

    return experiments


def main():
    full_plan = json.load(open('experiments_plan.json', 'r'))
    solve_plan(full_plan, [{}])

    new_experiments = {}
    for e in EXPERIMENT_AGGREGATOR:
        text = json.dumps(e, sort_keys=True)
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in new_experiments.keys():
            raise RuntimeError('sha256 collision! lol xD implement something to avoid this problem')
        new_experiments[h] = {'experiment_args': e}

    try:
        existing_experiments = json.load(open('experiments/jobs_list.json', 'r'))
    except FileNotFoundError:
        print(f'Creating new list with {len(new_experiments)} experiments.')
        json.dump(new_experiments, open('experiments/jobs_list.json', 'w'))
        return 0

    final_experiments = copy.deepcopy(existing_experiments)
    repeated = set(new_experiments.keys()).intersection(existing_experiments.keys())
    for key in repeated:
        new_experiments.pop(key)
        existing_experiments.pop(key)

    if len(new_experiments) == 0:
        print('No new experiments.')
        return 0

    if len(existing_experiments) > 0:
        raise NotImplementedError('Experiments unlisted in the plan are present. ABORTING')

    print(f'Adding {len(new_experiments)} new experiments.')
    final_experiments.update(new_experiments)
    json.dump(final_experiments, open('experiments/jobs_list.json', 'w'))


if __name__ == '__main__':
    main()
