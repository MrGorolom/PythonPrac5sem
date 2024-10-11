def get_new_dictionary(input_dict_name, output_dict_name):
    slovar = dict()
    f = open(input_dict_name)
    n = int(f.readline())
    for j in range(n):
        string = f.readline()
        key, values = string.split(' - ')
        if values[-1] == '\n':
            values = values[:-1]
        values = values.split(', ')
        for i in values:
            slovar.setdefault(i, list()).append(key)
    f.close()
    f = open(output_dict_name, 'w')
    n = len(slovar)
    f.write(str(n) + '\n')
    keys = sorted(slovar.keys())
    for i in range(n):
        f.write(keys[i] + ' - ' + ', '.join(sorted(slovar[keys[i]])) + '\n')
    f.close()
