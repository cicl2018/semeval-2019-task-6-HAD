with open('file2.tsv', 'w+') as w:
    with open('label-data.csv', 'r+') as r:
        for line in r.readlines():
            if len(line.split(",")) > 5:
                code = line.split(",")[0]
                off = line.split(",")[5]
                tweet = line.split(",")[6]
                
                if off == str(1) or off == 1:
                    off = "OFF"
                elif off == str(2) or off == 2:
                    off == "NOT"
                else:
                    off == "NOT"
                w.write(code + "\t" + tweet.replace("\n", "").replace("\r", "") + "\t" + off + "\tTIN\tOTH\n")
            # if  line.replace("\n", "").split("\t")[2] == "OFF":
            #     w.write(line)