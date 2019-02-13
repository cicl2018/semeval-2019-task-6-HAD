

# from nltk.tag import StanfordNERTagger
# from nltk.tokenize import word_tokenize


# st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
# 					   'stanford-ner.jar',
# 					   encoding='utf-8')


# text = ""

# with open ("dataset.txt", "r") as reading:
#     for line in reading.readlines():
#         # print (line.split("    ")[2])
#         save = line.split("    ")[2].replace("@", "").replace("(", "").replace(")", "")
#         text += save



# tokenized_text = word_tokenize(text.strip().replace("\n", " "))
# classified_text = st.tag(tokenized_text)

# with open ("combine.txt", "w+") as making:
#     for word in classified_text:
#         if not word[1] == 'O':
#             if word[1] == 'PERSON' or word[1] == 'ORGANIZATION': 
#                 if len(word[0]) > 2:
#                     making.write(word[0].lower() + "\n")


# from random import randint
# with open("test1.tsv", "w+") as wt:
#     with open("test_c.tsv", "r") as rt:
#         with open("pred.txt", "r") as rttext:
#             p = rttext.readlines()
#             lines = rt.readlines()
#             for line in range(len(lines)):
#                 pred = p[line].split("\t")
#                 save = lines[line].split("\t")
#                 wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "TIN\t" + pred[1])
                # if randint(0, 1) == 0:
                #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")
                # elif randint(0, 1) == 1:
                #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")
                # else:
                #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")   

with open("task_c.csv", "w+") as rt:
    with open("results.txt", "r") as rttext: 
        for line in rttext.readlines():
            values = line.split(" ")
            rt.write(values[0] + ", " + values[-1:][0].replace("\n", "") + "\n")



# with open("training_b.tsv", "r") as rd:
#     with open("training_1.tsv", "w+") as wr:
#         for line in rd.readlines():
#             if line.split("\t")[3].replace("\n", "") == "TIN":
#                 wr.write(line) 


# with open("top_OTH.txt", "r") as grp:
#     with open("oth.txt", "w+") as wg:
#         for line in grp.readlines():
#             # print " ".join(line.replace("\n", "").split("    ")[2].strip().split(" ")[:-1]).lower()
#             wg.write(" ".join(line.replace("\n", "").split("    ")[2].strip().split(" ")[:-1]).lower() + "\n")