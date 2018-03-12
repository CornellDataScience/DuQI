# DuQI: Duplicate Question Identification

Members: 
[Brandon Kates](https://github.com/BrandonKates),
[Zhao Shen](https://github.com/yuzhshen),
[Arnav Ghosh](https://github.com/garnav)

Objective: To create a system capable of detecting duplicate questions on 
Q&A platforms.

We expect our approach to help centralize the available knowledge on a single
question/issue and direct users with questions that have already been answered
to the appropriate resource.

We will test a variety of duplicate question identification methods on the 
[Quora question pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data),
and hope to eventually apply our findings to the classroom Q&A platform Piazza to
improve the Cornell student experience.

Data Requirements
=================

Below is the data required to successfully train/run all of the models.

In the current directory ("DuQI"), create a folder named "data" and populate it with:
- training and test data from [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data).

Final directory should look like:
- data
    - Quora training/test CSV files