# Semeval-2019-task-6-HAD
Evaluation of Offensive Tweets with target Classification. For more detail: [Coda Lab_OffensEval 2019 (SemEval 2019 - Task 6)](https://competitions.codalab.org/competitions/20011)

## Sub-tasks

Sub-task A - Offensive language identification;  <br/>
Sub-task B - Automatic categorization of offense types; <br/>
Sub-task C - Offense target identification.  <br/>

## Contributors 
**Himanshu Bansal** Universität Tübingen <br/>
**Daniel Nagel** Universität Tübingen <br/>
**Anita Soloveva**  Lomonosov MSU, Universität Tübingen <br/>

## Model

We are using Long short-term memory network (LSTM) as a main model.

## Pre-processing

1. Removing URLs and @USER <br/>
2. Parsing hashtags (See [Christos Baziotis et.al 2017](https://github.com/cbaziotis/ekphrasis))<br/>


