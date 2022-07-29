import pandas as pd
import scipy

import word2vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


Stopwords = stopwords.words('english')


#importing sentencetransformer 
from sentence_transformers import SentenceTransformer, util

from summarizer import Summarizer
# storing the bert summarizer model in a variable
bert_model = Summarizer()

# getting the model for keyphrase summary
model_ = SentenceTransformer('bert-base-nli-mean-tokens')

# getting model for keywords summary
model2 = SentenceTransformer('stsb-roberta-large')


''' getting summary from keyphrase '''

def keyphrase_abstractive_summary(text, keyphrases, get_summary, model_= model_):
  corpus=[]
  corpus=text.split('. ')
  corpus_embeddings = model_.encode(corpus)
  
  keyphrases_embeddings = model_.encode(keyphrases)
  for keyphrase, keyphrases_embedding in zip(keyphrases, keyphrases_embeddings):
    distances = scipy.spatial.distance.cdist([keyphrases_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    sentences = []
    for idx, distance in results:
        if(1-distance > 0.3):
            sentences.append(corpus[idx].strip())
    
            
#     creating a dataframe of sentences        
    df = pd.DataFrame({'text' : sentences})
    sentences=df.values.tolist()
    

    str1 = ' '.join([str(elem) for elem in sentences])
    summary = get_summary(str1)
  return(summary)


''' one keyword from a list of keywords'''
model_word2vec = word2vec.load('enwik9.bin')

def get_one_keyword(keyword_list):
  for i in range(len(keyword_list)):
    keywords = keyword_list[i]
    indexes, metrics = model_word2vec.analogy(pos=keywords, neg=[])
    keyword = model_word2vec.generate_response(indexes, metrics).tolist()[0][0]
    return(keyword)

''' getting text from keyword '''

def NLTK_proccessed(listrec):
#return string
    redundantchar=['@','#','!',"%",":",";","/","'",".",",","*","$","...","-","(",")"]
#stop words
    stop_words =set(stopwords.words("english"))
    #print(stop_words)
    stopsentences = []
    for sentence in listrec:
        stopfiltered_sentence= []
        tokenized_sentence=word_tokenize(sentence)
        for word in tokenized_sentence:
            if word not in stop_words and word not in redundantchar:
                stopfiltered_sentence.append(word)
        stopsentences.append(stopfiltered_sentence)
    
    
#stemming

    ps = PorterStemmer()
    stemmedsentences=[]
    for sentence in stopsentences:
        stemfiltered_sentence=[]
        for word in sentence:
            stemfiltered_sentence.append(ps.stem(word))
        stemmedsentences.append(stemfiltered_sentence)
    
#Lemmetization
    
    lem = WordNetLemmatizer()
    lemsentences=[]
    for sentence in stemmedsentences:
        lemfiltered_sentence=[]
        for word in sentence:
            lemfiltered_sentence.append(lem.lemmatize(word,"v"))
        lemsentences.append(lemfiltered_sentence)
    
#parts of speech (POS tagging)
    redundantpos=['CD','DT','NNS','NNP','NNPS','EX']
    possentences=[]
    for sentence in lemsentences:
        filteredpos_sentence=[]
        possentence = nltk.pos_tag(sentence)
        #print("pos", possentence)
        for record in possentence:
            if record[1] not in redundantpos:
                filteredpos_sentence.append(record[0])    
        possentences.append(filteredpos_sentence)    
    return (possentences)





def keyword_related_sentences(keyword, text):
    related_sentences=[]
    sentences=sent_tokenize(text)
    output_text=NLTK_proccessed(sentences)
    
    for m in range (len(keyword)):
        output_keyword=sent_tokenize(keyword[m])
    
    for i in range(len(output_text)):
      if(output_text[i] == []):
        cosine_scores=["0"]
      else:
        embedding1 = model2.encode(output_keyword[0], convert_to_tensor=True)
        embedding2 = model2.encode(output_text[i], convert_to_tensor=True)
        cosine_scores = (util.pytorch_cos_sim(embedding1, embedding2)).tolist()

      for j in range(len(cosine_scores)):
        for u in range(len(cosine_scores[j])):
          if (float(str(cosine_scores[j][u])) >= 0.4):
            related_sentences.append(sentences[i])
    str1 = ' '.join([str(elem) for elem in related_sentences])
    return(str1)