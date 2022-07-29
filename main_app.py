

import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
Stopwords = stopwords.words('english')
from nltk.stem import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('omw-1.4')

import pandas as pd
import os
import openai
openai.api_key = "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"



class SummarizerModel():
    
    def __init__(self):
        super().__init__()
        # self.initialized = False



    def inference(self, text):
        
        # getting response from openai for the text
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt="provide a since line summary of no more than 15 words from the following text:\n\n" + text + "\n\n",
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        
        return response.choices[0].text.strip()



    
    def get_topic_list(self, text, num_topics = 3):
        #     tokenizing in the sentences
            x = sent_tokenize(text)
            
        #     generating the list to store tokenized sentence
            sentences = []
            for line in x:
                        sentences.extend(sent_tokenize(line))
                    
        #     creating a dataframe of sentences        
            df = pd.DataFrame({'text' : sentences})
            
        #     vectorizing the sentences
            tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = tfidf.fit_transform(df['text'])
            
        #     using nmf model for topic modeling
            nmf_model = NMF(n_components=num_topics,random_state=42)
            nmf_model.fit(dtm)
            
        #     getting the topics separated
            topic_results = nmf_model.transform(dtm)
        #     print(topic_results)
            
        #     adding the topic numbers to the dataframe
            df['Topic'] = topic_results.argmax(axis=1)
            
        #     getting the list of unique topics
            topics = list(df.Topic.unique())
        #     print(topics)
            
        #     generating the list of differentiated topics
            b = []
            for topic in topics:
                b.append(' '.join(list(df[df.Topic == topic].text)))
                
        #     getting the list of sentences             
            return b



    '''getting topic modeling summaries'''
    def summary_on_topics(self, text):
        topics = self.get_topic_list(text)
        summaries = []
        for i, topic in enumerate(topics):
            summary = self.inference(topic)
            #summaries.append('0>')
            #summaries.append(str(i))
            summaries.append(summary)
            #summaries.append('       ')  
        df = pd.DataFrame(summaries, columns=['string_values'])
        return (df) 
    
    def NLTK_proccessed(self, listrec):
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
    
    def keyword_related_sentences(self, keyword, text):
        related_sentences=[]
        cosine_scores=[]
        sentences=sent_tokenize(text)
        output_text= self.NLTK_proccessed(sentences)
      
        #output_keyword=NLTK_proccessed(keywords)
        #print(output_keyword[0])
        for i in range(len(output_text)):
          #print(output_text[i])
          if(output_text[i] == []):
            cosine_scores=["0"]
          else:
            for x in range(len(output_text[i])):
              cosine = []
              n=wn.synsets(keyword)
              g=wn.synsets(output_text[i][x])
              
              sim = round(max((i.wup_similarity(n[0]) for i in g ), default=0), 1)
              
              cosine.append(sim)
              
          cosine_scores.append(cosine)
         
        for j in range(len(cosine_scores)):
          for u in range(len(cosine_scores[j])):
            if (float(str(cosine_scores[j][u])) >= 0.4):
              related_sentences.append(sentences[i])
        
                
    
        str1 = ' '.join([str(elem) for elem in related_sentences])
        return(str1)
    
    
       
    def get_one_keyword(self, keyword_list):
          #print(keyword_list)
          text_prompt = ' '.join(keyword_list)
          #print(text_prompt)
          response = openai.Completion.create(
          model="text-davinci-002",
          prompt="give me one keyword which relates to following three keywords:\n\nkeywords: " +text_prompt  + "\n\n",
          temperature=0.7,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0)
          #print(response)
          keyword = (((response.get("choices"))[0]).get("text"))[4:]
          #print(keyword)
          return(keyword)
    
    def keyword_summary(self, keys, text):
        #....
        keyword_list = keys
        #keyword_list.append(keys)
        one_key = self.get_one_keyword(keyword_list)
        related_sentences = self.keyword_related_sentences(one_key, text)
        #....
        summary = self.inference(related_sentences)
        return summary
        




# '''Getting the keywords related sentences'''

# ''' one keyword from a list of keywords'''
# model_word2vec = word2vec.load('enwik9.bin')
# @st.cache
# def get_one_keyword(keywordlist):
#   for i in range(len(keyword_list)):
#     keywords = keyword_list[i]
#     indexes, metrics = model_word2vec.analogy(pos=keywords, neg=[])
#     keyword = model_word2vec.generate_response(indexes, metrics).tolist()[0][0]
#     return(keyword)

# ''' getting text from keyword '''
# @st.cache
# def NLTK_proccessed(listrec):
# #return string
#     redundantchar=['@','#','!',"%",":",";","/","'",".",",","*","$","...","-","(",")"]
# #stop words
#     stop_words =set(stopwords.words("english"))
#     #print(stop_words)
#     stopsentences = []
#     for sentence in listrec:
#         stopfiltered_sentence= []
#         tokenized_sentence=word_tokenize(sentence)
#         for word in tokenized_sentence:
#             if word not in stop_words and word not in redundantchar:
#                 stopfiltered_sentence.append(word)
#         stopsentences.append(stopfiltered_sentence)
    
    
# #stemming

#     ps = PorterStemmer()
#     stemmedsentences=[]
#     for sentence in stopsentences:
#         stemfiltered_sentence=[]
#         for word in sentence:
#             stemfiltered_sentence.append(ps.stem(word))
#         stemmedsentences.append(stemfiltered_sentence)
    
# #Lemmetization
    
#     lem = WordNetLemmatizer()
#     lemsentences=[]
#     for sentence in stemmedsentences:
#         lemfiltered_sentence=[]
#         for word in sentence:
#             lemfiltered_sentence.append(lem.lemmatize(word,"v"))
#         lemsentences.append(lemfiltered_sentence)
    
# #parts of speech (POS tagging)
#     redundantpos=['CD','DT','NNS','NNP','NNPS','EX']
#     possentences=[]
#     for sentence in lemsentences:
#         filteredpos_sentence=[]
#         possentence = nltk.pos_tag(sentence)
#         #print("pos", possentence)
#         for record in possentence:
#             if record[1] not in redundantpos:
#                 filteredpos_sentence.append(record[0])    
#         possentences.append(filteredpos_sentence)    
#     return (possentences)




# @st.cache
# def keyword_related_sentences(keyword):
#     related_sentences=[]
#     sentences=sent_tokenize(text)
#     output_text=NLTK_proccessed(sentences)
    
#     for m in range (len(keyword)):
#         output_keyword=sent_tokenize(keyword[m])
    
#     for i in range(len(output_text)):
#       if(output_text[i] == []):
#         cosine_scores=["0"]
#       else:
#         embedding1 = model2.encode(output_keyword[0], convert_to_tensor=True)
#         embedding2 = model2.encode(output_text[i], convert_to_tensor=True)
#         cosine_scores = (util.pytorch_cos_sim(embedding1, embedding2)).tolist()

#       for j in range(len(cosine_scores)):
#         for u in range(len(cosine_scores[j])):
#           if (float(str(cosine_scores[j][u])) >= 0.4):
#             related_sentences.append(sentences[i])
#     str1 = ' '.join([str(elem) for elem in related_sentences])
#     return(str1)

# '''Getting the keyphrase related sentences'''