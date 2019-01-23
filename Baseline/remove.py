

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize


st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
					   'stanford-ner.jar',
					   encoding='utf-8')


text = ""

with open ("dataset.txt", "r") as reading:
    for line in reading.readlines():
        # print (line.split("    ")[2])
        save = line.split("    ")[2].lower().replace("@", "").replace("(", "").replace(")", "")
        text += save

# text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'


tokenized_text = word_tokenize(text.strip().replace("\n", " "))
classified_text = st.tag(tokenized_text)

print(classified_text)



# from random import randint
# with open("test1.tsv", "w+") as wt:
#     with open("test_b.tsv", "r") as rt:
#         lines = rt.readlines()
#         for line in range(len(lines)):
#             save = lines[line].split("\t")
#             print save
            # if randint(0, 1) == 0:
            #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")
            # elif randint(0, 1) == 1:
            #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")
            # else:
            #     wt.write(lines[line].replace("\n", "\t") + "OFF\t" + "UNT\t" + "UNI\n")   
