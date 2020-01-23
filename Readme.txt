----------------------------------------------------------------------
Readme - Predicting View Changes in Argumentative Discussions 
----------------------------------------------------------------------

======================================
About data
======================================

Data files are uploaded separately from code. Please unzip the files and put the folder under the root directory of the submitted code.

Explanation of the data files:

1. Lexicons as input files

    Under directory data_cmv/input/, each file contains the lexicon of a type of linguistic features as explained and cited in the supplementary pdf file. 


2. CMV discussion data

    Under directory data_cmv/, there are two files:
        cmv_threads.csv
        cmv_comments.csv

    cmv_threads.csv contains the discussion level data and the original post (OP) from an opinion holder (OH), and cmv_comments.csv contains the comment level data. The first line of each file contains the column names. These two files can be cross referenced by the thread_id.


======================================
About code
======================================
    Under the root directory, there are notebooks:
        linearCRF.ipynb
        skipCRF.ipynb

    , and a file contains dependencies (packages and modules):
        requirement.txt 

    The code is a notebook created in Google Colab. One snippet of code need to be added at the beginning to mount Google drive and install the required modules before executing the code on Colab. We will transform the code into .py format upon publication.

 





