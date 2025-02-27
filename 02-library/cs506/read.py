import csv
import os

def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """
    data = []
    with open(csv_file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row_data = [float(x) if x.isnumeric() else x for x in row]
            data.append(row_data)
    return data
