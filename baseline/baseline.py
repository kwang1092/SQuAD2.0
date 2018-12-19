import json
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

training_json = {}
development_json = {}
test_json = {}

####Dictionaries for each data set
training_dict = {}
development_dict = {}
test_dict = {}

####TRAINING PREPROCESSING
with open('training.json','r') as f:
    training_json = json.load(f)

for x in training_json['data']:
    for p in x['paragraphs']:
        context = p['context']
        questions = {}
        for q in p['qas']:
            qid = q['id']
            # questions[qid] = {q['question']:q['is_impossible']}
            questions[qid] = q['question']
        if context not in training_dict:
            training_dict[context] = questions
        else:
            training_dict[context].update(questions)


####DEVELOPMENT PREPROCESSING
with open('development.json','r') as f:
    development_json = json.load(f)

for x in development_json['data']:
    for p in x['paragraphs']:
        context = p['context']
        questions = {}
        for q in p['qas']:
            qid = q['id']
            # questions[qid] = {q['question']:q['is_impossible']}
            questions[qid] = q['question']
        if context not in development_dict:
            development_dict[context] = questions
        else:
            development_dict[context].update(questions)


####TESTING PREPROCESSING
with open('test.json','r') as f:
    test_json = json.load(f)

for x in test_json['data']:
    for p in x['paragraphs']:
        context = p['context']
        questions = {}
        for q in p['qas']:
            qid = q['id']
            questions[qid] = q['question']
        if context not in test_dict:
            test_dict[context] = questions
        else:
            test_dict[context].update(questions)


################ BASELINES ##################
# Baseline 1 : Compare question to entire context. Determine % of words
# from question that appear in context. Classify question as category
# 0 or 1.
#
# Baseline 2 : Compare question to context sentences. Determine % of words
# from question that appear in sentence. Classify question as category
# 0 or 1.
#
# Baseline 3 : Compare question to context sentences. Determine % of nouns
# from question that appear in sentence. Classify question as category
# 0 or 1.

############### BASELINES TESTING ##################
##BASELINE1

with open ('baseline1.csv', mode = 'w') as f:

    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Id','Category'])

    for x in test_dict:
        xLower = x.lower()
        for q in test_dict[x]:
            answer = 0
            questionLower = test_dict[x][q].lower()
            questionWords = questionLower.split(' ')
            includedWords = 0
            totalWords = 0
            for word in questionWords:
                if word in xLower:
                    includedWords += 1
                totalWords += 1
            if includedWords/totalWords >= 0.5:
                answer = 1

            writer.writerow([q,answer])


#####BASELINE 2
with open ('baseline2.csv', mode = 'w') as f:

    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Id','Category'])

    for x in test_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in test_dict[x]:
            answer = 0
            questionLower = test_dict[x][q].lower()
            #print(questionLower)
            questionWords = questionLower.split(' ')
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word in sentence:
                        includedWords += 1
                    totalWords += 1
                #print(includedWords)
                #print(totalWords)
                #print(includedWords/totalWords)
                if includedWords/totalWords >= 0.5:
                    answer = 1

            writer.writerow([q,answer])

##BASELINE3

def process(question):
    question = nltk.word_tokenize(question)
    question = nltk.pos_tag(question)
    return question

with open ('baseline3.csv', mode = 'w') as f:

    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['Id','Category'])
    nouns = ['NNS', 'NNP', 'NN', 'NNPS']

    for x in test_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in test_dict[x]:
            answer = 0
            questionLower = test_dict[x][q].lower()
            #print(questionLower)
            questionWords = process(questionLower)
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word[1] in nouns:
                        if word[0] in sentence:
                            includedWords += 1
                        totalWords += 1
                #print(includedWords)
                #print(totalWords)
                if totalWords != 0 and includedWords/totalWords >= .5:
                    answer = 1
                    #print(includedWords/totalWords)

            writer.writerow([q,answer])

# print('done with testing baselines')
############## BASELINES TRAINING ##################
#BASELINE1

with open ('baseline1tr.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])
    jsonDict = {}

    for x in training_dict:
        xLower = x.lower()
        for q in training_dict[x]:
            answer = 0
            questionLower = training_dict[x][q].lower()
            questionWords = questionLower.split(' ')
            includedWords = 0
            totalWords = 0
            for word in questionWords:
                if word in xLower:
                    includedWords += 1
                totalWords += 1
            if includedWords/totalWords >= .5:
                answer = 1

            jsonDict[q] = answer
            # writer.writerow([q,answer])
    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)

# print('done with training baseline 1')
####BASELINE 2
with open ('baseline2tr.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])
    jsonDict = {}

    for x in training_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in training_dict[x]:
            answer = 0
            questionLower = training_dict[x][q].lower()
            #print(questionLower)
            questionWords = questionLower.split(' ')
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word in sentence:
                        includedWords += 1
                    totalWords += 1
                #print(includedWords)
                #print(totalWords)
                #print(includedWords/totalWords)
                if includedWords/totalWords >= .5:
                    answer = 1

            jsonDict[q] = answer
            # writer.writerow([q,answer])
    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)

# print('done with training baseline 2')
##BASELINE3

with open ('baseline3tr.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])

    jsonDict = {}

    nouns = ['NNS', 'NNP', 'NN', 'NNPS']

    for x in training_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in training_dict[x]:
            answer = 0
            questionLower = training_dict[x][q].lower()
            #print(questionLower)
            questionWords = process(questionLower)
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word[1] in nouns:
                        if word[0] in sentence:
                            includedWords += 1
                        totalWords += 1
                #print(includedWords)
                #print(totalWords)
                if totalWords != 0 and includedWords/totalWords >= .5:
                    answer = 1
                    #print(includedWords/totalWords)

            jsonDict[q] = answer
            # writer.writerow([q,answer])
    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)


# print('done with training baseline 3')
################ BASELINES development ##################
###BASELINE1

with open ('baseline1val.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])
    jsonDict = {}

    for x in development_dict:
        xLower = x.lower()
        for q in development_dict[x]:
            answer = 0
            questionLower = development_dict[x][q].lower()
            questionWords = questionLower.split(' ')
            includedWords = 0
            totalWords = 0
            for word in questionWords:
                if word in xLower:
                    includedWords += 1
                totalWords += 1
            if includedWords/totalWords >= .5:
                answer = 1

            jsonDict[q] = answer
            # writer.writerow([q,answer])
    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)

# print('done with val baseline 1')
#####BASELINE 2
with open ('baseline2val.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])
    jsonDict = {}

    for x in development_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in development_dict[x]:
            answer = 0
            questionLower = development_dict[x][q].lower()
            #print(questionLower)
            questionWords = questionLower.split(' ')
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word in sentence:
                        includedWords += 1
                    totalWords += 1
                #print(includedWords)
                #print(totalWords)
                #print(includedWords/totalWords)
                if includedWords/totalWords >= .5:
                    answer = 1

            jsonDict[q] = answer
            # writer.writerow([q,answer])

    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)
# print('done with val baseline 2')
##BASELINE3

with open ('baseline3val.json', mode = 'w') as f:

    # writer = csv.writer(f, delimiter = ',')
    # writer.writerow(['Id','Category'])
    jsonDict = {}

    nouns = ['NNS', 'NNP', 'NN', 'NNPS']

    for x in development_dict:
        xLower = x.lower()
        xLowerSplit = xLower.split('.')
        #print(xLowerSplit)

        for q in development_dict[x]:
            answer = 0
            questionLower = development_dict[x][q].lower()
            #print(questionLower)
            questionWords = process(questionLower)
            #print(questionWords)

            for sentence in xLowerSplit:
                includedWords = 0
                totalWords = 0
                for word in questionWords:
                    #print(word)
                    if word[1] in nouns:
                        if word[0] in sentence:
                            includedWords += 1
                        totalWords += 1
                #print(includedWords)
                #print(totalWords)
                if totalWords != 0 and includedWords/totalWords >= 0.5:
                    answer = 1
                    #print(includedWords/totalWords)

            jsonDict[q] = answer
            # writer.writerow([q,answer])

    jsonDict = str(jsonDict)
    jsonDict = jsonDict.replace("'", '"')
    f.write(jsonDict)
# print('done with val baseline 3')
