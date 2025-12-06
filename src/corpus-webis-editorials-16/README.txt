README FOR THE WEBIS-EDITORIAL-16 CORPUS, VERSION 1.


This folder contains all files of the Webis-Editorials-16 corpus. In case you published any results related to the corpus, please cite the following paper:

@INPROCEEDINGS{alkhatib:2016,
    AUTHOR = {Khalid Al-Khatib and Henning Wachsmuth and Johannes Kiesel and Matthias Hagen and Benno Stein},
    TITLE  = {{A News Editorial Corpus for Mining Argumentation Strategies}},
    YEAR   = {2016},
    BOOKTITLE = {Proceedings of the 26th International Conference on Computational Linguistics (COLING 16)},
    PAGES = {3433--3443},
    LOCATION = {Osaka, Japan},
}


You will find the following folders and files here:


================
./annotated-txt/
================

This folder contains all annotated editorials in plain text format. Each given TXT file stores one editorial in a tab-separated form. 

In particular, each line in the file corresponds to one argumentative discourse unit (or a paragraph separator) and consists of three tab-separated entries:

  - Number of the unit/separator (ordered increasingly)
  - Label of the unit/separator
  - Text of the unit (empty in case of a separator)

The corpus is provided in two splits, given in the two following subfolders:

--------------------------------
./annotated-txt/split-by-portal/
--------------------------------

This split contains three subfolders, one for each news portal contained in the corpus (aljazeera, foxnews, guardian). 

Each subfolder contains 100 editorials.

-------------------------------------
./annotated-txt/split-for-evaluation/
-------------------------------------

This split contains three subfolders, one that represents the training set, one that represents the validation set (development set), and one that represents the test set (held-out set). 

The editorials are split by the time they were last updated. In particular, the training set consists of the first 180 editorials, the validation set of the next 60 editorials, and the test set the last 60 editorials.


================
./annotated-xmi/
================

This folder contains all annotated editorials in the Apache UIMA format. Each given XMI file stores one editorial together with all annotations. 

General information about the XMI format can be found at http://uima.apache.org. The type system that defines the used annotation scheme is found in the folder ./uima-type-systems/ described below.

Unlike the plain text format above, the annotations include all metadata given for the editorials in their sources. This metadata is described at the very end of this file.

As for the plain text format, the corpus is provided in two splits, given in the two following subfolders:

--------------------------------
./annotated-txt/split-by-portal/
--------------------------------

This split contains three subfolders, one for each news portal contained in the corpus (aljazeera, foxnews, guardian). 

Each subfolder contains 100 editorials.

-------------------------------------
./annotated-txt/split-for-evaluation/
-------------------------------------

This split contains three subfolders, one that represents the training set, one that represents the validation set (development set), and one that represents the test set (held-out set). 

The editorials are split by the time they were last updated. In particular, the training set consists of the first 180 editorials, the validation set of the next 60 editorials, and the test set the last 60 editorials.


====================
./uima-type-systems/
====================

This folder contains the Apache UIMA type systems that are need to process the above-mentioned XMI files. 

In particular, the type system to be used is stored in the file "ArgumentationTypeSystem.xml". Notice, though, that this type system imports some of the other given type systems, so keep them all together.



================
unnannotated.csv
================

This tab-separated CSV file contains all unnannotated editorials together with their metadata. 

In particular, each line of the file corresponds to one editorial and consists of the following tab-separated entries:

  - A unique ID 
  - The name of the portal the editorial refers to
  - The date the editorial was last updated 
  - The source URL of the editorial	
  - The name(s) of the author(s) of the editorial 
  - The title of the editorial 
  - The text of the editorial 
  - A short summary of the editorial

As mentioned, the metadata can also be found in the XMI files described above.




