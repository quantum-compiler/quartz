from quartz import PyAction


def DecodePyActionList(action_list: [PyAction]):
    decoded_actions = []
    for action in action_list:
        decoded_actions.append((action.qubit_idx_0, action.qubit_idx_0))
    return decoded_actions
