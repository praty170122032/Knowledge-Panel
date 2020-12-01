# Knowledge-Panel
A semantic search query based implementation to develop a knowledge panel for normalized entities extracted by NER engines using IBM Watson Discovery service.

- Developed API to query and analyze different corpora on IBM Watson Discovery service.
- Built HTML parser named entity recognition engine to extract entities from the corpus and generated ground truth to evaluate its precision & recall.
- Performed normalization on entities and extracted their abbreviations from Wikipedia summary using web scraping, to develop a knowledge panel for semantic search.

Listed are the different situations and challenges/tasks that I came across during my intern project and how I overcame/completed them in *STAR* format.


- IBM Watson Discovery service was used to create and store databases online.
- Watson Discovery service enables you to rapidly ingest, normalize, enrich, search and analyze by querying your unstructured data.
- Worked on developing API to query and analyze different corpora on IBM Watson Discovery service.


## Comparison of NER engines
- The next task was to compare the performance of discovery NER engine which the company was using to extract entities with its types with other possible NER engines that can be developed using python libraries.
- So, I developed 2 different NER engines using Spacy and Regular expression libraries respectively.
The 3 different NER’s worked as follows – 
- *Discovery* – Discovery service enriches the unstructured data that we upload automatically and creates a JSON file for each document containing all the enrichment. This NER engines directly queried the JSON files for entity extraction.
- *Spacy* – Spacy is an important NLP application library. It automatically extracts the entities from a text provided and labels it to its predefined types. 
- *HTML Naviga* – Company had a preprocessed format of the data in which they had added the HTML parsed text into the JSON of each document. So, this NER used regular expression to extract each entity and its types in HTML tags from the HTML text of documents.

- The task was to generate the ground truth and compare the performance of the 3 NER engines and find which is a better source for entity extraction.
*TASK:* 
- To perform a statistical analysis of extracted entities, generate ground truth table and plot graphs of recall and precision calculated from them.
*Action:*
- Target was to create a table with voted ground truth and all the entities with their document id and text from which they are extracted. For ground truth, votes were collected, and according to them, I created a function to assign the correct ground truth for an entity. With each doc id, I queried Naviga1 from API to get the associated text and appended the respective columns.


## Normalization
*Challenge* –  The next challenge that came up was the extracted entities had too many matched groups of entities, like for example – Mr. Obama, President Obama, Barack Obama,etc are different entities but they represent the same person or meaning. Hence, normalization was required on extracted entities.

*Action-*
Made a General Script File for importing the required functions and applied on each of location, company and person entities to get a normalized dictionary. Did manual post processing for each entity type before getting normalised terms. Using various matching algorithms, extracted normalized term for each matched group by defined set of rules, and their respective abbreviations from Wikipedia summary, to develop a knowledge panel for entities.


## Abbreviation Extraction
*Situation -* 
- To extract abbreviations for all normalized terms of types Location and Organization.

*Task -* 
- To find and extract abbreviations for all normalized terms of types Location and Organization from Wikipedia summary, and create respective dictionaries.

*Action -* 
- Extracted the summary of wiki page for each entity through web scraping using beautiful soup and converted the HTML to plain text.
- Defined 3 different methods to extract all possible abbreviations from the summary, using schwartz_hearst abbreviation extraction and regular expressions.
- Pulled first letters of the normalized term, and used jaro winkler matching algorithm to get the matching score of each extracted abbreviation for that term.
- Using defined cut off, assigned the abbreviation having maximum score to the normalized term in the dictionary.
