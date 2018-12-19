# SQuAD2.0
Our attempt to determine if questions are answerable with a given context paragraph for the SQuAD2.0 dataset

Preprocessing
We parsed the json files for the training, development, and test sets and built nested
dictionaries for each set that kept track of the context, question, question id, and other
information that we deemed important (we stored different information during different
iterations of the models as well as the final model). While we were comparing the question and
the context, we eliminated the question mark from each question, and we made the entirety of the
question and the context lowercase as we wanted to avoid discrepancies in capitalization as
capitalization doesn’t typically affect whether a question can or cannot be answered. Each of our
baselines involves categorizing the question based on a percentage (explained for each model
below); for every model, we tried multiple percentage values (referred to as % cut off below) to
see how the model’s performance was impacted

Baseline 1
For each word in the question, we checked if that word appeared in the context. If the
percentage of words from the question appeared in the context was above a certain
cut-off, we assigned the question a label of 0 or 1 (not answerable, answerable).

Baseline 2
For each question, we looked at each sentence in the corresponding context, and we
determined what percentage of words from the question appeared in the sentence. Based
on the highest percentage that was determined, we assigned the question a label of 0 or 1.
For example, if a question’s context had three sentences, and the percentage cut off was
not met for two of the sentences and it was met for one of the sentences, a 1 would be
assigned as the label.

Baseline 3
We used nltk to assign a part of speech to each word in the question. For each question,
we looked at each sentence in the corresponding context, and we calculated the
percentage of the nouns from the question that appeared in the sentence. This allowed us
to ignore all “filler” words and only look at the “important” words as nouns are often the
focus/target. Based on the highest percentage determined, we assigned the question a
label of 0 or 1. We used nltk as shown here:
https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88
e7da

Final System:

Preprocessing
Similar to the baselines, we parsed through the json files for each data json set. While
parsing through each context, we used spaCy to count the number of each named entity type in
the context and added those counts to a feature vector array that was maintained for each unique
question id. For each question, specifically, we tokenized the sentence and counted whether or
not each question type indicator appeared (when, how, why, etc…) and appended those to the
same feature vector. Finally, we used nltk to tag each word in the question’s part of speech
(noun, pronoun, and whether it was a number or not). Using this count, we then kept track of the
percentage of each one of these that appeared in the question that also appeared in the context
and appended all three of these percentages to the end of the feature vector.
Following, we also used spaCy to identify the root of the question, stemmed it, and
checked whether or not this stemmed root was also a stemmed root of any one of the sentences in
the context via spaCy’s clustering and parsing tree functions. We appended this boolean to the
end of the feature vector as well and maintained a dictionary of question id keys with the feature
vector as the value.

Implementation and motivation
For our final system, we decided to take a machine learning approach so as to be better able to
utilize the training and development set and create a model that would be better able to determine
question answerability than our heuristic models. We created a feature vector for each question,
then trained each feature vector using logistic regression. We chose features in the following
categories.

1. # of named entities found in the context: for the context corresponding to the question,
we counted the number of each type of named entity found. For example, if there are a lot
of people found in the context, and the question starts with “who”, it is likely that the
question is answerable.

2. Question type: determining the question type for each question would give us a hint on
whether or not a question could be answerable, given a particular context. For example, if
the answer type is a location, but the context does not provide any locations, then it is
likely that the question would not be answerable. We created one-hot encoding features
for each of the following keywords (0 if word doesn’t exist in question, 1 if it does): who,
what, when, where, why, how, many.

3. Percentage of nouns, pronouns, and numbers from question found in context: we
decided to check what percentage of each nouns, pronouns, and numbers from each
question can be found in the context. For example, if none of the nouns found in the
question could be found in the context, then it is likely that the question would not be
answerable.

4. Compare focus of question to the focus of each sentence in the context: we created a
parse tree for the question and each sentence in the context. We then compared the
(stemmed) root of the question, to the (stemmed) roots/subroots of each sentence, and
checked if the root of the question exists as one of the roots/subroots of any sentence in
the context. If the focus is found, then the feature has value 1; 0 otherwise.

5. Sentence embedding (SE) of question vs.sentence embedding for each sentence in
the context: we wanted to determine if there is a sentence in the context that is similar to
the focus of the question. To do this, we decided to find the SE of the question, and
compare it to each sentence from the context’s SE. We calculated the cosine similarity
between the question SE and each sentence’s SE, and made the maximum cosine
similarity one of the features.

We utilized the following packages for our implementation:
1. SpaCy: named entity recognition, parse tree
2. NLTK: parse the questions into words and context into sentences.
3. Sci-kit learn: used to run logistic regression
4. InferSent: create sentence embeddings
We used the following site to create a parse tree and look for the root:
https://towardsdatascience.com/building-a-question-answering-system-part-1-9388aadff507?fbcl
id=IwAR3y1jQ4j-u2cf85gMkg5yRTXrtveidW94D8UF93fE5ZDejucVPjdCdFtog

Final Model - Variation 1 (Feature types 1, 2, 3)
We included the following features: # of each NER type in context, question type, % of
nouns, pronouns, and numbers from question found in context

Final Model - Variation 2 (Feature types 1 (modified), 2, 3)
This model is a similar to variation 1, with a slight modification. For the feature type 1,
we considered that some of the NER types could be combined or eliminated because they
didn’t occur very often.
We chose to combine some NER types to create the following categories:
1) Person
2) Nationalities/religious/political groups + companies/institutions
3) Countries/cities + locations/landmarks
4) Buildings + objects + languages
5) Date + time
6) Percent + money + quantity + misc. Numerals
We got rid of the following categories: events, works of art, law, ordinal

Final Model - Variation 3 (Feature types 1, 2)
We just eliminated feature type 3 from final model variation 1. We considered that the
percentage of nouns, pronouns, and numbers would throw off the model.

Final Model - Variation 4 (Feature types 1 (modified), 2, 3, 4)
We didn’t have time to run this variation on the development set. Variation 4 used the
mixed NERs as seen in Variation 2 along with all the other features and the root boolean.




