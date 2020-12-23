from pathlib import Path

# Converts IDs to ids from 0 to N vertices

# ID to id
ID = {}

# id to ID
vertexCount = 0

# filenames
data_name = "mon100"
data_dir = f"../data/mont/labelled/{data_name}"
in_filepath = f"../data/mont/{data_name}.csv"
out_filepath = f"{data_dir}/data.txt"
label_filepath = f"{data_dir}/label.txt"

Path(data_dir).mkdir(parents=True, exist_ok=True)

delimiter = ","
with open(in_filepath, "r") as in_file, \
        open(out_filepath, "w") as out_file, \
        open(label_filepath, "w") as label_file:
    for i, line in enumerate(in_file):
        split = line.split(delimiter)
        id1 = int(split[0])
        id2 = int(split[1])
        # print("line {}: {} {}".format(i, id1, id2))

        if id1 not in ID:
            ID[id1] = vertexCount
            v1 = vertexCount
            vertexCount += 1
            label_file.write(f"{id1}\n")
        else:
            v1 = ID[id1]

        if id2 not in ID:
            ID[id2] = vertexCount
            v2 = vertexCount
            vertexCount += 1
            label_file.write(f"{id2}\n")
        else:
            v2 = ID[id2]
        out_file.write(f"{v1} {v2}\n")
