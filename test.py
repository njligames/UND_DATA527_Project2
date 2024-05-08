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

def loadFile():
    filename = "input_fl_12477"
    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    fields = []
    mydict = []
    for line in lines:
        row = line.split()
        if [] == fields:
            fields = row
        else:
            item = {}
            for i in range(len(fields)):
                item[fields[i]] = row[i]
            mydict.append(item)
    
    with open(filename + ".csv", 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(mydict)

class TestCalculations(unittest.TestCase):

    def test_load(self):
        loadFile()


if __name__ == '__main__':
    unittest.main()

