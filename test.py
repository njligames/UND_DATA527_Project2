# project/test.py

import unittest

import csv

def loadRValues(path):
    r_values = None
    if os.path.isfile(path):
        with open(path, 'r') as openfile:
            r_values = json.load(openfile)
    if None == r_values:
        r_values = []


    return r_values

def appendRValue(path, r_values, val):
    r_values.append(val)
    with open(path, "w") as outfile:
        json.dump(r_values, outfile)

def normalizeMinMaxScaling(ary):
    if len(ary) > 0:
        _max = max(ary)
        _min = min(ary)
        new_ary = []
        for item in ary:
            val = float(item)
            if _min == _max:
                new_ary.append(1.0)
            else:
                new_ary.append((val-_min)/(_max-_min))
        return new_ary
    return ary

def writeFile(filename, fields, mydict):
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(mydict)

def shouldAdd(field, fields):
    for f in fields:
        if field == f:
            return True
    return False

def parseFields(lines, special_fields):
    fields = []
    mydict = []

    field_dict = {}
    for line in lines:
        row = line.split()
        if [] == fields:
            for field in row:
                fields.append(field)

            for field in fields:
                field_dict[field] = []
        else:
            ary_item = []
            item = {}
            for i in range(len(fields)):
                ary_item.append(row[i])
                item[fields[i]] = row[i]
                field_dict[fields[i]].append(float(row[i]))
            mydict.append(item)
    return field_dict, fields, mydict

def loadFile(fields_in, fields_out):

    filename = "input_fl_12477"
    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    field_dict, fields, mydict = parseFields(lines, fields_in)
    writeFile(filename + ".preProcess.csv", fields, mydict)

    n = 0
    field_in_dict = {}
    field_in = []
    field_out_dict = {}
    field_out = []
    for i in range(len(fields)):
        field_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])

        if shouldAdd(fields[i], fields_in):
            field_in_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])
            n = len(field_in_dict[fields[i]])
            field_in.append(fields[i])
        if shouldAdd(fields[i], fields_out):
            field_out_dict[fields[i]] = normalizeMinMaxScaling(field_dict[fields[i]])
            n = len(field_out_dict[fields[i]])
            field_out.append(fields[i])

    X = []
    Y = []
    i = 0
    while i < n:
        item = []
        for field in field_in:
            item.append(field_in_dict[field][i])
        X.append(item)

        item = []
        for field in field_out:
            item.append(field_out_dict[field][i])
        Y.append(item)

        i = i + 1

    mydict = []
    for i in range(n):
        item = {}
        for j in range(len(fields)):
            item[fields[j]] = field_dict[fields[j]][i]
        mydict.append(item)
    writeFile(filename + ".postProcess.csv", fields, mydict)

    mydict = []
    for i in range(n):
        item = {}
        for j in range(len(field_in)):
            item[field_in[j]] = field_in_dict[field_in[j]][i]
        mydict.append(item)
    writeFile(filename + ".X.csv", field_in, mydict)

    mydict = []
    for i in range(n):
        item = {}
        for j in range(len(field_out)):
            item[field_out[j]] = field_out_dict[field_out[j]][i]
        mydict.append(item)
    writeFile(filename + ".Y.csv", field_out, mydict)

    return X, Y

class TestCalculations(unittest.TestCase):

    def test_load(self):
        fields_in = ["altitude", "indicated_airspeed", "roll"]
        fields_out = ["pitch"]
        X, Y = loadFile(fields_in, fields_out)
        print(X)
        print(Y)


if __name__ == '__main__':
    unittest.main()

