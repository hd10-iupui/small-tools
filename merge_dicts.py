def merge_dict(x, y):
    # print(x)
    # print(y)
    for k, v in x.items():
        if k in y.keys():
            y[k] = y[k] + v
        else:
            y[k] = v
    # print(y)


def merge_merge_dict(x, y):
    for km, vm in x.items():
        if km in y.keys():
            merge_dict(vm, y[km])
        else:
            y[km] = vm
