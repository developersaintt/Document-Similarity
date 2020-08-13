# Intro 

this is a document similarity checker I made out of curiosity about how the plaigrism checker works but in the end I knew they don't work how I thought they work. So here is what I coded. 😁

## Requirements
library      | version 
------------ | -------------
Numpy        |    1.16.2
NLTK         |    3.4.4
Pandas       |    0.24.2
Textract     |    1.6.3

you can install any missing library using the following command...

```javascript
pip install <library_name>
```

## line_similarity.py

this file takes two strings as an argument and will tell you the similarity between them...

```javascript
python line_similarity -l1 "this is line number one" -l2 "this is line number two"
```
for example the similarity between these two lines 
> Indeed, Iran should be put on notice that efforts to try to remake Iraq in their image will be aggressively put down" he said.

> Iran should be on notice that attempts to remake Iraq in Iran\'s image will be aggressively put down, he said

is 97.53%


## document_similarity.py
this file takes two file path as argument and a threshold and will give at the output the lines that are similar above the threshold ...

```javascript
python document_similarity.py -f1 "path/to/first/docfile.docx" -f2 "path/to/second/docfile.docx" -t 0.8
```
