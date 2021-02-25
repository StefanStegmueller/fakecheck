GermanTRC: Corpus Description


=====================================================================

Description of the .json format:

For every ID, there are upto three components:
	[A] Fake news information (1)
	[B] True news information (1 or more)
	[C] Crawled True news information (0 or more)

 ---------------------------
| [A] Fake news information |
 ---------------------------
This component contains the following information of the news article containing fake news:

Date  : publication date of the article
URL   : URL of the website
Title : Headline of the article 
Text  : Body plain-text of the article

A maximum of three false statements are provided:
False_Statement_1 : False claim 1
False_Statement_2 : False claim 2
False_Statement_3 : False claim 3

Ratio_of_Fake_Statements : total number of fake claims, range [1:3] 

Overall_Rating of the disinformation in text : range [0.1:1.0]
0.1 no disinformation in text 
0.2
0.3
0.4
0.5 neutral / ambivalent 
0.6
0.7
0.8
0.9
1.0 strong disinformative text 

 ---------------------------
| [B] True news information |
 ---------------------------
This component contains information of one or more news articles that refutes the corresponding false statement(s). For each news article, the following information is provided:

Date  : publication date of the article
URL   : URL of the website
Title : Headline of the article 
Text  : Body plain-text of the article

True_Statement_1: True statement / facts.

 -----------------------------------
| [C] Crawled True news information |
 -----------------------------------
This component contains information of zero or more crawled news articles related to the topic of corresponding fake news. For each crawled news article, the following information is provided:

URL   : URL of the website
Title : Headline of the article 
Text  : Body plain-text of the article

======================================================================
The original sources retain the copyright of the data.

You are allowed to use this dataset for research purposes only.

For more question about the dataset, please contact:
XXXX, XXXX 

v1.0 01/12/2020

