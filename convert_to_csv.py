import os
from sys import call_tracing

input_folder = './OID/Dataset'
datasets = ["train", "test", "validation"]

def convert_dir(name):
    output = open("output.csv", "w")
    for s in datasets:
        for i, file in enumerate(os.listdir(os.path.join( input_folder, s, name, "Label"))):
            csv = open(s.path.join( input_folder, s, name, "Label"), "r")
            for l in csv:
                type_det = ""
                parts = l.split(",")
                if(parts[0] == "Car"):
                    type_det = "car"
                else:
                    type_det = "lp"
                output.write(s.upper() + ",./imgs/" + s + "/" + file[:-3] + "jpg," +  type_det  )

