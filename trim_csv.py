import os

train_csv = open("./OID/csv_folder/train-annotations-bbox.csv")
validation_csv = open("./OID/csv_folder/validation-annotations-bbox.csv")
test_csv = open("./OID/csv_folder/test-annotations-bbox.csv")
csvs = [train_csv, validation_csv, test_csv]
output = open("output.csv", "w")
folders = ["train", "validation", "test"]
i = 0
for csv in csvs:
    x= 0
    for l in csv:
        parts = l.split(',')
        annotation = ""
        if x > 1:
            break
        if parts[2] == "/m/0k4j":
            annotation = "car"
        elif parts[2] == "/m/01jfm_":
            annotation = "lp"
            x += 1
        else:
            continue
        
        output.write(folders[i].upper() +",./imgs/" +folders[i] + "/" + parts[0] + ".jpg," + annotation +"," + parts[4]+ "," + parts[6]+ ",,," + parts[5]+ "," + parts[7] + ",,\n")

    i += 1