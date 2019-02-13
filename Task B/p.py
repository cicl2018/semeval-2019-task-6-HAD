with open('training1.tsv', 'w+') as w:
    with open('training.tsv', 'r+') as r:
        for line in r.readlines():
            if  line.replace("\n", "").split("\t")[2] == "OFF":
                w.write(line)