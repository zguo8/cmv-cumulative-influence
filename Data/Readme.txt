----------------------------------------------------------------------
Readme - Predicting View Changes in Argumentative Discussions 
----------------------------------------------------------------------
This the a README for the supplementary materials of submission # 1410 - Predicting View Changes in Argumentative Discussions.

======================================
About data
======================================

Data files are submitted separately from code. Please unzip the files and put the folder under the root directory of the submitted code.

Explanation of the data files:

1. Lexicons as input files

    Under directory data_cmv/input/, each file contains the lexicon of a type of linguistic features as explained and cited in the supplementary pdf file. 


2. CMV discussion data

    Under directory data_cmv/, there are two files*:
        cmv_threads.csv
        cmv_comments.csv

    cmv_threads.csv contains the discussion level data and the original post (OP) from an opinion holder (OH), and cmv_comments.csv contains the comment level data. The first line of each file contains the column names. These two files can be cross referenced by the thread_id.

* note: due to the file size limitation, we provide only sample data cmv_threads_sample.csv and cmv_comments_sample.csv. The complete dataset will be available  upon publication.




