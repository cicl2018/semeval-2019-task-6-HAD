# SemEval-2019-task-6-HAD
Evaluation of Offensive Tweets with target Classification. For more details: [Coda Lab_OffensEval 2019 (SemEval 2019 - Task 6)](https://competitions.codalab.org/competitions/20011)

## Sub-tasks

### Sub-task A - Offensive language identification (Offensive / Not Offensive) <br/>
 - **15 Jan 2019:** A test data release - **17 Jan 2019:** Submission deadline <br/>
### Sub-task B - Automatic categorization of offense types (Targeted Insult and Threats / Untargeted) <br/> 
- **22 Jan 2019:** A test data release - **24 Jan 2019:** Submission deadline <br/>
### Sub-task C - Offense target identification (Target: Individual / Group / Other)<br/>
 - **29 Jan 2019:** A test data release - **31 Jan 2019:** Submission deadline  <br/>

## Contributors 
**Himanshu Bansal** Univesity of Tübingen himanshu.bansal@student.uni-tuebingen.de <br/>
**Daniel Nagel** University of Tübingen daniel.nagel@student.uni-tuebingen.de <br/>
**Anita Soloveva**  Lomonosov MSU, University of Tübingen anita.soloveva@student.uni-tuebingen.de <br/>

## Preprocessing
1. Lowercasing <br/>
2. Removing URLs, @USER, all the following charachters  “ :. , — ˜ ”, digits and single quotation marks except for abbreviations and possessors (e.g. u’re → u’re, but about’ → about) <br/>
3. Using ‘=’, ‘!’, ‘?’ and ‘/’ as token splitters  (e.g. something!important → something important) <br/>
4. Parsing hashtags (See [Christos Baziotis et. al. 2017](https://github.com/cbaziotis/ekphrasis))<br/>

## Model
We are using an LSTM based classifier
### Sub-task A: Approaches
[1](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20A/Task_A_only_Preprocessing.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-test.tsv))  <br/>
[2](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20A/Task_A_Badword_list.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-test.tsv))  + Postprocessing with manually created offensive word list <br/>
[3](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20A/Task_A_hashtag_parsing.py). Parsing hashtags + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-test.tsv))
### Sub-task B: Approaches
[1](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20B/Task_B_only_Preprocessing.py).  All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-b-test.tsv)) <br/>
[2](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20B/Task_B_Badword_list.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-b-test.tsv))  + Postprocessing with manually created  database of potential insult victims as targets. A large part is [the names of representatives of top twitter profiles from the USA, the UK, Saudi Arabia, Brazil, India and Spain, Iran, Iraq, Turkey, Russia and Germany](https://www.socialbakers.com/statistics/twitter/profiles/).

### Sub-task C: Approaches
[1](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20C/Task_C_only_Preprocessing.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-c-test.tsv))  <br/>
[2](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20C/Task_C_Badword_list.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-c-test.tsv))  + Postprocessing with manually created  database of potential insult victims as targets, which split by categories: (IND), (GRP) and (OTH). <br/>
[3](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/daniel/Task%20C/Task_C_Badword_list.py). All preprocessing steps + LSTM model (architecture parameters are optimized by [SVM predictions](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/svm-predictions-c-test.tsv))  + Postprocessing with manually created  database, (see Sub-task C: 2) and personal pronouns, including their contractions. 
###  All datasets can be issued via mail.


