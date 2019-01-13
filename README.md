# Semeval-2019-task-6-HAD
Evaluation of Offensive Tweets with target Classification. For more detail: [Coda Lab_OffensEval 2019 (SemEval 2019 - Task 6)](https://competitions.codalab.org/competitions/20011)

## Sub-tasks

Sub-task A - Offensive language identification; <br/>
 - **15 Jan 2019:** A test data release - **17 Jan 2019:** Submission deadline <br/>
Sub-task B - Automatic categorization of offense types; <br/> 
- **22 Jan 2019:** A test data release - **24 Jan 2019:** Submission deadline <br/>
Sub-task C - Offense target identification. <br/>
 - **29 Jan 2019:** A test data release - **31 Jan 2019:** Submission deadline  <br/>

## Contributors 
**Himanshu Bansal** Universität Tübingen <br/>
**Daniel Nagel** Universität Tübingen <br/>
**Anita Soloveva**  Lomonosov MSU, Universität Tübingen <br/>

## Preprocessing

1. Removing URLs and @USER <br/>
2. Parsing hashtags (See [Christos Baziotis et. al. 2017](https://github.com/cbaziotis/ekphrasis))<br/>

## Our approaches

1. We are using [Long short-term memory network (LSTM) model](https://github.com/cicl2018/semeval-2019-task-6-HAD/blob/master/Baseline/script.py). <br/>
2. LSTM with fasttext vectors <br/>
### Sub-task A
1. Using an additional preprocessed training set of tweets <br/>
2. Postprocessing with emojis set and offensive word list
### Sub-task B && C
1. Using a list of ethnic slurs
2. Using a list of [top twitter profiles from United States, United Kindom, Saudi Arabia, Brazil and Spain] (https://www.socialbakers.com/statistics/twitter/profiles/)


