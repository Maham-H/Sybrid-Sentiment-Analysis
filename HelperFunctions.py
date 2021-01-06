import csv
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import re


from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer("english", ignore_stopwords=True)

#nltk.download('stopwords')

####################################################################################

####################################################################################


def load_data(filename, num_annotate=1,shift=True):
    ###
    # Inputs:
    #    # filename: (string: csv file name)
    #    # num_annotate: (integer: 1 or 2) determines if one or two sets of labels are present
    #    # shift: tells if the 1st line of .csv file is heading
    # Outputs:
    #    #Comments, labels and brandname
    # 
    ###
    reader = csv.reader(open(filename, 'r'), delimiter= ",")
    Tweets,Labels1,Labels2,brand = read_output(reader,num_annotate,shift)
    
    if num_annotate == 2:
        return Tweets,Labels1,Labels2,brand
    else:
        return Tweets,Labels1,brand
        
####################################################################################

####################################################################################

def tokenizer(data):
    return data.split()

def tokenizer_porter(text,porter):
    return [porter.stem(word) for word in text.split()]
        
####################################################################################

####################################################################################

def category_tokenization(Tweets, Labels,all_stopwords,porter,stop_words=False,tokenize_porter=False):
    tokens = {
        "all":[],
        "tok_vp" : [],
        "tok_p" : [],
        "tok_ne" : [],
        "tok_n" : [],
        "tok_vn" : [],
        "tok_q":[],
    }
    count=0
    for t in Tweets:
        if tokenize_porter:
            tok_temp = tokenizer_porter(t,porter)#word_tokenize(t)
        else:
            tok_temp = tokenizer(t)
        if stop_words:
            tok_temp = [word for word in tok_temp if not word in all_stopwords]
        tokens["all"] += tok_temp
        if Labels[count]=='Very Positive':
            tokens["tok_vp"] += tok_temp
        elif Labels[count]=='Positive':
            tokens["tok_p"] += tok_temp
        elif Labels[count]=='Neutral':
            tokens["tok_ne"] += tok_temp
        elif Labels[count]=='Negative':
            tokens["tok_n"] += tok_temp
        elif Labels[count]=='Very Negative':
            tokens["tok_vn"] += tok_temp
        elif Labels[count]=='Query':
            tokens["tok_q"] += tok_temp
        else:
            pass
        count+=1
    return tokens

####################################################################################

####################################################################################

def most_common(tokens):
    c= Counter(tokens["all"])
    c_vp= Counter(tokens["tok_vp"])
    c_p= Counter(tokens["tok_p"])
    c_ne= Counter(tokens["tok_ne"])
    c_n= Counter(tokens["tok_n"])
    c_vn= Counter(tokens["tok_vn"])
    c_q= Counter(tokens["tok_q"])

    print("Most common tokens in all vocabulary: \n",c.most_common(10),'\n')
    print("Most common tokens in VeryPositive: \n",c_vp.most_common(10),'\n')
    print("Most common tokens in Positive: \n",c_p.most_common(10),'\n')
    print("Most common tokens in Neutral: \n",c_ne.most_common(10),'\n')
    print("Most common tokens in Negative: \n",c_n.most_common(10),'\n')
    print("Most common tokens in VeryNegative: \n",c_vn.most_common(10),'\n')
    print("Most common tokens in Query: \n",c_q.most_common(10),'\n')
    return c,c_vp,c_p,c_ne,c_n,c_vn,c_q
    
####################################################################################

####################################################################################

def read_output(reader,num_annotate,shift):
    Labels2=[]
    Tweets=[]
    Labels1=[]
    brand=[]
    num_tweets=0
    for line in reader:
        count=0
        for field in line:
            count +=1
            if count==1:
                brand.append(field)
            elif(count==2):
                Tweets.append(field)
            elif (count==3):
                Labels1.append(field)
            elif (count==4) and num_annotate==2:
                Labels2.append(field)
                
        num_tweets+=1
    if num_tweets>0 and shift==True:
        brand = brand[1:np.size(Tweets)]
        Tweets = Tweets[1:np.size(Tweets)]
        Labels1 = Labels1[1:np.size(Labels1)]
        if num_annotate==2:
            Labels2 = Labels2[1:np.size(Labels2)]
    return Tweets,Labels1,Labels2,brand
        
####################################################################################

####################################################################################


def store_data(filename,Tweets_cleaned,Labels,brand):
    # Storing to csv File
    count = np.size(Labels)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Brand','Comments', 'Labels'])
        for c in range(count):
            writer.writerow([brand[c],Tweets_cleaned[c], Labels[c]])
    file.close()

####################################################################################

####################################################################################

def category_reduction(cm):

    c11= cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    c12= cm[0][2]+cm[0][3]+cm[1][2]+cm[1][3]
    c13= cm[0][4]+cm[0][5]+cm[1][4]+cm[1][5]


    c21= cm[2][0]+cm[2][1]+cm[3][0]+cm[3][1]
    c22= cm[2][2]+cm[2][3]+cm[3][2]+cm[3][3]
    c23= cm[2][4]+cm[2][5]+cm[3][4]+cm[3][5]

    c31= cm[4][0]+cm[4][1]+cm[5][0]+cm[5][1]
    c32= cm[4][2]+cm[4][3]+cm[5][2]+cm[5][3]
    c33= cm[4][4]+cm[4][5]+cm[5][4]+cm[5][5]


    c = [[c11, c12,c13],
         [c21, c22,c23],
         [c31, c32,c33]]

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(121)
    cax = ax.matshow(c,cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix of predictions')

    N2= np.shape(c)[0]
    thresh=900
    for i in range(N2):
        for j in range(N2):
            plt.text(j, i, "{:,}".format(c[i][j]),
            horizontalalignment="center",
            color="White" if c[i][j] > thresh else "black")

    labels=['Positive', 'Neutral','Negative']
    plt.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('predictions')
    plt.ylabel('actual labels')
    plt.show()
    
    accuracy = (c11+c22+c33)/np.sum(c)
    print('Accuracy : ', accuracy*100)
    p_o = np.trace(np.asarray(c))
    
    #observed_agreement_rate
    observed_agreement_rate = p_o/np.sum(c)
    # random_chance_agreement_rate
    p_e = np.sum(np.sum(c,0)*np.sum(c,1))/np.sum(c) 

    kappa = (p_o - p_e)/(np.sum(c)-p_e)
    print('Observed Agreement :', p_o)
    print("Random Chance Agreement :", p_e)
    print("Kappa :", kappa)
    
    return c, accuracy

####################################################################################

####################################################################################
        
def plot_results(preds,labels,y_test):


    cm = metrics.confusion_matrix(y_test, preds, labels)
    conf_mat = cm

    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat,cmap=plt.get_cmap('Greens'))
    plt.title('Confusion matrix of predictions')
    
    thresh = 0.001#maxcm/0.1
    
    
    label =np.unique(y_test,return_counts=False)

    N = np.size(labels);
    for i in range(N):
        for j in range(N):
            plt.text(j, i, "{:,}".format(conf_mat[i][j]),
            horizontalalignment="center",
            color="black" if conf_mat[i][j] > thresh else "black")

    plt.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('predictions')
    plt.ylabel('actual labels')
    plt.show()
    return conf_mat

####################################################################################

####################################################################################

def clean_data(Tweets,all_stopwords):
    count = 0

    Tweets_cleaned = []

    # rules for elements being eliminated from the tweets
    r_at = r'@[A-Za-z0-9_]+'  # Removing @ from all tweets
    r_hash = r'#[A-Za-z0-9_]+' # Removing hash tags
    r_rt = r'RT '  # Removing RT i.e. if the tweet is a retweet
    r_emoji = '[^a-zA-Z]'  # Removing emoji and replacing with space
    #r_brandtag=r'#'+brand.lower()
    vectorizer = TfidfVectorizer()
    tokens=[]
    r_se = r'[:]]'
    r_se2 = r'[=)]'
    r_se3 = r'[:-D]'
    r_se4 = r'[:D]'
    r_se5 = r'[=D]'
    r_se6 = r'[:)]'

    r_sae2 = r'[:(]'
    r_sae3 = r'[:[]'
    r_sae4 = r'[=(]'
    
    r_ae = r'[>:(]'
    r_ae2 = r'[>:(]'
    
    r_le = r'[(y)]'
    r_le2 = r'[(Y)]'
    
    for t in Tweets:
        
        
        clean_tweets = re.sub(r'|'.join((r_at, r_rt)),'',t)  
        
        clean_tweets = re.sub('https?:[A-Za-z0-9./]+','URL',clean_tweets)
        #clean_tweets = re.sub(r_brandtag,'Brandtag ',clean_tweets)
        #clean_tweets = re.sub(r''+brand,'Brandtag ',clean_tweets)
        #clean_tweets = re.sub(r'|'.join((r_se,r_se2,r_se3,r_se4,r_se5,r_se6)),'happy',clean_tweets)
        #clean_tweets = re.sub(r'|'.join((r_sae2,r_sae3,r_sae4)),'sad',clean_tweets)
        #clean_tweets = re.sub(r'|'.join((r_ae,r_ae2)),'angry',clean_tweets)
        #clean_tweets = re.sub(r'|'.join((r_le,r_le2)),'like',clean_tweets)
        #clean_tweets = re.sub(r'[<3]','love',clean_tweets)
        clean_tweets = re.sub(r_hash, ' ', clean_tweets)
        clean_tweets = re.sub(r_emoji, ' ', clean_tweets)
        clean_tweets = re.sub('[\s ]+', ' ',clean_tweets)
        clean_tweets = clean_tweets.lower()
    

        

        Tweets_cleaned.append(clean_tweets)
        tokens+=tokenizer_porter(clean_tweets,porter)
    tokens = [word for word in tokens if not word in all_stopwords]


    vectorizer.fit_transform(Tweets_cleaned)

    #print(vectorizer.idf_)
    #print(vectorizer.vocabulary_)
    return Tweets_cleaned#, tokens


####################################################################################

####################################################################################

def count_labels(Labels1, Labels2):
    
    cm ={"":0,
         "Very Negative" : 0,
         "Negative": 0,
         "Neutral" : 0,
         "Positive" : 0,
         "Very Positive" : 0,
         "Query":0,
         
         "Very NegativeNeutral" :0,
         "Very NegativePositive" : 0,
         "Very NegativeVery Positive" :0,
         "Very NegativeNegative" : 0,
         "Very NegativeQuery" : 0,
         
         "NegativeNeutral" :0,
         "NegativePositive" : 0,
         "NegativeVery Positive" :0,
         "NegativeVery Negative" : 0,
         "NegativeQuery" : 0,
         
         "NeutralVery Negative" :0,
         "NeutralNegative" : 0,
         "NeutralVery Positive" :0,
         "NeutralPositive" : 0,
         "NeutralQuery" : 0,
         
         "PositiveNeutral" :0,
         "PositiveNegative" : 0,
         "PositiveVery Positive" :0,
         "PositiveVery Negative" : 0,
         "PositiveQuery" : 0,
         
         "Very PositiveNeutral" :0,
         "Very PositiveNegative" : 0,
         "Very PositivePositive" :0,
         "Very PositiveVery Negative" : 0,
         "Very PositiveQuery" : 0,
         
         "QueryVery Positive" :0,
         "QueryNegative" : 0,
         "QueryPositive" :0,
         "QueryVery Negative" : 0,
         "QueryNeutral" : 0,
         
        }
    try: 
        assert np.size(Labels1)==np.size(Labels2)
    except:
        print('Labels don\'t have same size')
        return cm
    N = np.size(Labels1)
    for i in range(N):
        if Labels1[i]==Labels2[i]:
            try:
                cm[Labels1[i]] += 1
            except:
                pass
                #print(Labels1[i])
        else:
            #if Labels1[i]!='Query' and Labels2[i]!='Query':
            cm[Labels1[i]+Labels2[i]] +=1
        conf_mat =[
            [cm["Very Negative"], cm["Very NegativeNegative"], 
                cm["Very NegativeNeutral"], cm["Very NegativeQuery"], cm["Very NegativePositive"],
               cm["Very NegativeVery Positive"]],
               
               [cm["NegativeVery Negative"], cm["Negative"], 
                cm["NegativeNeutral"],cm["NegativeQuery"], cm["NegativePositive"],
               cm["NegativeVery Positive"]],
               
               [cm["NeutralVery Negative"], cm["NeutralNegative"], 
                cm["Neutral"],cm["NeutralQuery"], cm["NeutralPositive"],
               cm["NeutralVery Positive"]],
            
                [cm["QueryVery Negative"], cm["QueryNegative"], 
                cm["QueryNeutral"], cm["Query"] , cm["QueryPositive"],
               cm["QueryVery Positive"]],
               
               [cm["PositiveVery Negative"], cm["PositiveNegative"], 
                cm["PositiveNeutral"],cm["PositiveQuery"], cm["Positive"],
               cm["PositiveVery Positive"]],
               
               [cm["Very PositiveVery Negative"], cm["Very PositiveNegative"], 
                cm["Very PositiveNeutral"],cm["Very PositiveQuery"], 
                cm["Very PositivePositive"],cm["Very Positive"]]
                   
                
              ]
    
    labels = ['VeryPositive', 'Positive', 'Neutral','Query', 'Negative', 'VeryNegative']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat,cmap=plt.get_cmap('Greens'))
    plt.title('Confusion matrix of labelled annotations')
    
    maxcm=max(max(conf_mat))
    thresh = maxcm/0.1
    
    
    #label =np.unique(Labels1,return_counts=False)
    N = np.size(labels);
    for i in range(N):
        for j in range(N):
            plt.text(j, i, "{:,}".format(conf_mat[i][j]),
            horizontalalignment="center",
            color="white" if conf_mat[i][j] > thresh else "black")


    plt.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Annotator 2')
    plt.ylabel('Annotator 1')
    plt.show()
    #observed_agreement
    p_o = np.trace(np.asarray(conf_mat))
    #observed_agreement_rate
    observed_agreement_rate = p_o/np.sum(conf_mat)
    # random_chance_agreement_rate
    p_e = np.sum(np.sum(conf_mat,0)*np.sum(conf_mat,1))/np.sum(conf_mat) 
    
    kappa = (p_o - p_e)/(np.sum(conf_mat)-p_e)
    print('Observed Agreement :', p_o)
    print("Random Chance Agreement :", p_e)
    print("Kappa :", kappa)
    #label, count =np.unique(Labels,return_counts=True)
    #cm =[count[5],count[3],count[2],count[1],count[4]]
    return cm
        
####################################################################################

####################################################################################

def return_stopwords(lists=[],append=True):
    all_stopwords = stopwords.words("english")
    if append:
        all_stopwords.append('brandtag')
        all_stopwords.append('url')
        all_stopwords.append('season')
        all_stopwords.append('series')
        all_stopwords.append('review')
        all_stopwords.append('seri')
        all_stopwords.append('ki')
        all_stopwords.append('ha')
        all_stopwords.append('ka')
        all_stopwords.append('ke')
        all_stopwords.append('hai')
        all_stopwords.append('k')
        all_stopwords.append('ko')
        all_stopwords.append('hain')
        all_stopwords.append('ho')
        all_stopwords.append('se')
        all_stopwords.append('ye')
        all_stopwords.append('bhi')
        all_stopwords.append('mein')
        all_stopwords.append('koi')
        all_stopwords.append('kia')
        all_stopwords.append('b')
        all_stopwords.append('ya')
        all_stopwords.append('yeh')
        all_stopwords.append('ab')
        all_stopwords.append('hi')
        all_stopwords.append('aur')
        all_stopwords.append('hy')
        all_stopwords.append('kya')
        all_stopwords.append('h')
        all_stopwords.append('is')
        all_stopwords.append('he')
        all_stopwords.append('hi')
        all_stopwords.append('to')
        all_stopwords.append('and')
        all_stopwords.append('or')
        all_stopwords.append('i')
        all_stopwords.append('me')
    if lists!=[]:
        for i in range(np.size(lists)):
            all_stopwords.append(lists[i])
    return all_stopwords