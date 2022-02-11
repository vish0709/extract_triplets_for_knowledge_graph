import spacy
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import re
import time
import ast

# Put the paragraph in text
text=""" A snake charmer is a person who moves the streets with different types of the banks of the river Yamuna. It is snakes in his basket. He goes from one place to another to show various types of snakes and their tricks. """



dfa = {0:{'0':0, '1':1},
       1:{'0':2, '1':0},
       2:{'0':1, '1':2}}

# #Ram works in Delhi
# {0:Subject, 1:Predicate, 2:Object} 
rule1 = {0:{'NOUN':0, 'ADJ':0, 'ADV':0, 'ADP':0, 'AUX':1, 'CCONJ':0, 'DET':0, 'INTJ':0, 'NUM':0, 'PART':0, 'PRON':0, 'PROPN':0, 'PUNCT':0, 'SCONJ':0, 'SYM':0, 'VERB':1, 'X':0,"SPACE":0},
       1:{'NOUN':2, 'ADJ':1, 'ADV':1, 'ADP':1, 'AUX':1, 'CCONJ':1, 'DET':2, 'INTJ':1, 'NUM':1, 'PART':1, 'PRON':1, 'PROPN':2, 'PUNCT':1, 'SCONJ':1, 'SYM':1, 'VERB':1, 'X':2,"SPACE":1},
       2:{'NOUN':2, 'ADJ':2, 'ADV':2, 'ADP':2, 'AUX':1, 'CCONJ':2, 'DET':2, 'INTJ':1, 'NUM':2, 'PART':1, 'PRON':1, 'PROPN':2, 'PUNCT':2, 'SCONJ':2, 'SYM':1, 'VERB':1, 'X':2,"SPACE":2}}

def accepts(transitions,initial,accepting,s,words):
    state = initial
    prevState = initial
    sub,subWord = [],[]
    pred,predWord = [],[]
    obj,objWord = [],[]
    for i in range(len(s)):
        if prevState == 0 and transitions[state][s[i]] == 0:
            sub.append(s[i])
            subWord.append(str(words[i]).strip())
        elif (prevState == 1 and transitions[state][s[i]] == 1) or (prevState == 0 and transitions[state][s[i]] == 1):
            pred.append(s[i])
            predWord.append(str(words[i]).strip())
        elif (prevState == 1 and transitions[state][s[i]] == 2) or (prevState == 2 and transitions[state][s[i]] == 2):
            obj.append(s[i])
            objWord.append(str(words[i]).strip())
        # print('-------->>>>',prevState,c,'----->>',transitions[state][c],'-------->>>',transitions[state])
        state = transitions[state][s[i]]
        prevState = state
    return state in accepting, sub, pred, obj, subWord,predWord,objWord


nlp = spacy.load("en_core_web_sm")

def rdfCreator(text):
    text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
    text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
    doc = nlp(text)
    # sentences = [str(sent) for sent in doc.sents]
    # sentences = [sent for sent in doc.]
    word,posTag = [],[]
    for sentence in doc.sents:
        tempWord = []
        tempPosTag = []
        for token in sentence:
            tempWord.append(token)
            tempPosTag.append(token.pos_)
        word.append(tempWord)
        posTag.append(tempPosTag)
            # print(token, token.pos_)
        # print("*"*70)
    return word,posTag

def extract_rdf(words,posTags):
    rdf = {}
    prev_sub = ""
    prev_pred = ""
    prev_obj = ""
    finalTokens = []
    isRdf,sub,pred,obj,subWord,predWord,objWord = False,[],[],[],[],[],[]
    start = 0
    end = 3
    prevStatus = False
    lastObj = []
    lastEnd = 0
    c=0
    while True:
        # print("Inside Outer while")
        # print(words[start:end])
        if len(posTags) == end-1:
            break
        isRdf,sub,pred,obj,subWord,predWord,objWord=accepts(rule1,0,{2},posTags[start:end],words[start:end])
        end+=1

        if isRdf:
            s = 0
            e = len(sub)-1
            while True:
                if len(sub)==0 or len(obj)==0:
                    break
                if sub[s] in ["CCONJ","ADP","DET","ADV","PRON"] :
                    sub.pop(s)
                    subWord.pop(s)
                    e-=1
                elif sub[e] in ["CCONJ","ADP","DET","ADV"]:
                    sub.pop(e)
                    subWord.pop(e)
                    e-=1 
                else:
                    break
            if len(sub)==0 or len(obj)==0:
                continue
            prevStatus = True
            rdf[str(subWord)] = (predWord,objWord)
            lastObj = obj
            lastEnd = end
            while True:
                # print("Inside Inner while")
                # print(words[start:end])
                isRdf,sub,pred,obj,subWord,predWord,objWord=accepts(rule1,0,{2},posTags[start:end],words[start:end])
                if isRdf:
                    s = 0
                    s_obj = 0
                    e = len(sub)-1
                    e_obj = len(obj)-1
                    while True:
                        if len(sub)==0 or len(obj)==0:
                            break
                        if sub[s] in ["CCONJ","ADP","DET","ADV","PRON"] :
                            sub.pop(s)
                            subWord.pop(s)
                            e-=1
                        elif sub[e] in ["CCONJ","ADP","DET","ADV"]:
                            sub.pop(e)
                            subWord.pop(e)
                            e-=1
                        elif obj[s_obj] in ["CCONJ","ADP","DET","ADV"]:
                            obj.pop(s_obj)
                            objWord.pop(s_obj)
                            e_obj-=1
                        elif obj[e_obj] in ["CCONJ","ADP","DET","ADV"]:
                            obj.pop(e_obj)
                            objWord.pop(e_obj)
                            e_obj-=1
                        else:
                            break
                    if len(sub)==0 or len(obj)==0:
                        continue
                    rdf[str(subWord)] = (predWord,objWord)
                    lastObj = obj
                    lastEnd = end
                else:
                    break
                if len(posTags) == end-1:
                    break
                end+=1
        else:
            if prevStatus:
                start = lastEnd-len(lastObj)
                end = end+2
        # time.sleep(1)
        if c==40:
            break
        c+=1
    return rdf

words,posTag = rdfCreator(text)
subject = []
predicate = []
obj = []
verbose = True
progress = tqdm(desc='Extracting Entity and Relationships: ', total=len(words)) if verbose else None
for i in range(len(words)):
    # print(words[i],"--------------",posTag[i])
    # print(extract_rdf(words[i],posTag[i]))
    rdf = extract_rdf(words[i],posTag[i])
    for k,v in rdf.items():
        # print(type(k),'---->>',k.replace("[",'').replace("]",'').split(","))
        # print(type(v[0]),'------>',v[0])
        sub = k.replace("[",'').replace("]",'').replace("'",'').split(",")
        subject.append(" ".join(sub))
        predicate.append(" ".join(v[0]))
        obj.append(" ".join(v[1]))
    progress.update(1) if verbose else None

progress.close() if verbose else None

# print("*"*80)
df = pd.DataFrame()

df['Subject'] = subject
df['Predicate'] = predicate
df['Object'] = obj

print(df)


import networkx as nx
import matplotlib.pyplot as plt

def draw_kg(pairs):
    k_graph = nx.from_pandas_edgelist(pairs, 'Subject', 'Object',
            create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white',
        )
    labels = dict(zip(list(zip(pairs.Subject, pairs.Object)),
                  pairs['Predicate'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()

draw_kg(df)
