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
**Himanshu Bansal** Univesity of Tübingen <br/>
**Daniel Nagel** University of Tübingen <br/>
**Anita Soloveva**  Lomonosov MSU, University of Tübingen <br/>

## Preprocessing
1. Lowercasing <br/>
2. Removing URLs, @USER, all the following charachters  “ :. , — ˜ ”, digits and single quotation marks except for abbreviations and possessors (e.g. u’re → u’re, but about’ → about) <br/>
3. Using ‘=’, ‘!’, ‘?’ and ‘/’ as token splitters  (e.g. something!important → something important) <br/>
4. Parsing hashtags (See [Christos Baziotis et. al. 2017](https://github.com/cbaziotis/ekphrasis))<br/>

## Model
We are using an LSTM based classifier
### Sub-task A: Approaches
1. All preprocessing steps + LSTM model (architecture parameters are optimized by SVM predictions)  <br/>
2. All preprocessing steps + LSTM model (architecture parameters are optimized by SVM predictions)  + Postprocessing with manually created offensive word list <br/>
3. Parsing hashtags + LSTM model (architecture parameters are optimized by SVM predictions)
### Sub-task B: Approaches
1.  All preprocessing steps + LSTM model (architecture parameters are optimized by SVM predictions) <br/>
2. All preprocessing steps + LSTM model (architecture parameters are optimized by SVM predictions)  + Postprocessing with manually created  database of potential insult victims as targets (can be issued). A large part is [names of representatives of top twitter profiles from the USA, the UK, Saudi Arabia, Brazil, India and Spain, since these countries have the most Twitter users and Iran, Iraq, Turkey, Russia and Germany, because we predicted a possible aggression towards the users from these countries](https://www.socialbakers.com/statistics/twitter/profiles/).

### Sub-task C: Approaches
1. Using a list of ethnic slurs
2. Using a list of [top twitter profiles from United States, United Kindom, Saudi Arabia, Brazil and Spain] (https://www.socialbakers.com/statistics/twitter/profiles/)


