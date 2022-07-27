import streamlit as st
from PIL import Image
import main_app


technique = main_app.SummarizerModel()


logo = Image.open('logo.png')
col1, col2 = st.columns([1, 9])
col1.image(logo, 60, 60)
col2.title('Discite Text Generator')


main_text= st.container()


main_text.subheader('Paste your input text here')

# with main_text.form(key="second", clear_on_submit=False):
input_text = main_text.text_area('Please input your text here:')  

    # submitted = st.form_submit_button("Submit")
    # if submitted:
    #     st.write('text length: ', len(text_t5))

Keywords = st.container()
Keywords.subheader('Provide some keywords')
one, two, three = st.columns(3)
    
key1 = one.text_input('Keyword 1')
key2 = two.text_input('Keyword 2')
key3 = three.text_input('Keyword 3')

# Keyphrase_ = st.container() 
# Keyphrase_.subheader('Provide a replative keyphrase here')   
# keyphrase = Keyphrase_.text_input('Put some keyphrase here')


output = st.container()
main_output = st.container()
# abstractive_output, extractive_output = st.columns(2)
abstractive_output = st.container()
extra_abs_output = st.container()
topic_output = st.container()
keywords_output = st.container()
# keyphrase_output = st.container()




    



waiting_text = 'waiting for text...'

with main_output.form(key="forth", clear_on_submit=False):
    text1 = text2 = text3 = text4 = text5 = text6 = waiting_text

    if input_text:
        waiting_text = waiting_text 
        if not key1 or not key2 or not key3:
            text6 = 'waiting for keywords'
        else:
            keys = [key1, key2, key3]
            text6 = ' '.join(keys)
        if not keyphrase:
            text4 = waiting_text
        else: 
            text4 = 'Summary text'
        text1 = technique.inference(input_text)
        text2 = 'Summary text'
        text3 = technique.summary_on_topics(input_text)
        text5 = 'Summary text'
        
    
    
        

        
    abstractive_output.text_area(label="Abstractive output:", value=text1, height=200)
    # extractive_output.text_area(label="Extractive output:", value=text2, height=200)
    topic_output.text_area(label="Topic output:", value=text3, height=100       )
    # keyphrase_output.text_area(label="Keyphrase related output:", value=text4, height=20)
    # extra_abs_output.text_area(label="Extractive Abstractive related output:", value=text5, height=20)
    # keywords_output.text_area(label="Keywords related output:", value=text6, height=20)
        
        
    
    
    
    
    
    # if not text_t5:
    #     waiting_text = waiting_text
    # elif text_t5 and choose_model == 'CNN':
    #     if not key1 or not key2 or not key3:
    #         text6 = 'waiting for some keyword'
    #     else:
    #         keys = [key1, key2, key3]
    #         text6 = 'some text with CNN'
    #     if not keyphrase:
    #         text4 = waiting_text
    #     else: 
    #         text4 = 'Some keyphrase related text with CNN'
    #     text1 = 'something with CNN'
    #     text2 = 'something with CNN'
    #     text3 = 'Some Topic with CNN'
    #     text5 = 'something with CNN'
        
        
    # elif text_t5 and choose_model == 'T5':
    #     if not key1 or not key2 or not key3:
    #         text6 = 'waiting for keywords'
    #     elif key1 and key2 and key3:
    #         keys = [key1, key2, key3]
    #         text6 = 'something with T5'  
    #     if not keyphrase:
    #         text4 = waiting_text
    #     else: 
    #         text4 = 'Some keyphrase related text with T5'
    #     text1 = 'Something with T5'
    #     text2 = 'something with T5'
    #     text3 = 'Somethine with T5'
    #     text5 = 'something with T5'
    
    
    # elif text_t5 and choose_model == 'Pegasus':
    #     if not key1 or not key2 or not key3:
    #         text6 = 'waiting for keywords'
    #     else:
    #         keys = [key1, key2, key3]
    #         text6 = ' '.join(keys)
    #     if not keyphrase:
    #         text4 = waiting_text
    #     else: 
    #         text4 = 'Some keyphrase related text with Pegasus'
    #     text1 = 'something with Pegasus'
    #     text2 = 'something with Pegasus'
    #     text3 = 'Some Topic with Pegasus'
    #     text5 = 'something with Pegasus'
        
    
    
    
    
    
    
    # first, second, third = st.columns(3)
    # num_topics = first.text_input("Please input the number of topics here:")

    #submitted = st.form_submit_button("Generate")

    # if submitted:
        
        #with st.spinner('Generating Summary...'):
         #   abstractive_summary =  get_summary(st.session_state.text)
          #  st.subheader("Summary")
           # st.write(abstractive_summary)
            
            
            
#model_t5 = T5ForConditionalGeneration.from_pretrained('t5-small')
#tokenizer_t5 = T5Tokenizer.from_pretrained('t5-small')


#st.cache()
#def get_summary(text, model = model_t5 , tokenizer = tokenizer_t5 , min_length=30, max_length=50):
#  summarizer = pipeline("summarization", model=model, tokenizer=tokenizer_t5 , framework="tf")
#  summary = summarizer(text, min_length, max_length)
#  summary_text = summary[0]['summary_text']
#  return summary_text
#fin


# with one.form(key="keyword1", clear_on_submit=False):
#     key1 = one.text_area('keyword1')
#     one.session_state.text=key1
    # submitted = st.form_submit_button("Submit")
    # if submitted:
    #     st.write('keyword 1: ', key1 )
    

# with two.form(key="keyword2", clear_on_submit=False):
    # key2 = two.text_area('keyword2')
    # two.session_state.text=key2
    # submitted = st.form_submit_button("Submit")
    # if submitted:
    #     st.write('keyword 2: ', key2 )
    

# with three.form(key="keyword3", clear_on_submit=False):
    # key3 = three.text_area('keyword3')
    # three.session_state.text=key3
    # submitted = st.form_submit_button("Submit")
    # if submitted:
    #     st.write('keyword 3: ', key3 )





# with st.sidebar:
       
#     choose_model = st.radio("Kindly choose the model for summarization", ('CNN', 'T5', 'Pegasus'))
    
#     with st.spinner('Waiting ...'):
#         if not text_t5:
        
#             if choose_model == 'CNN':
#                 time.sleep(1)
#                 output.text('CNN model has been chosen')
                
            
#             elif choose_model == 'Pegasus':
#                 time.sleep(1)
#                 output.text('Pegasus model has been chosen')
        
        
#             elif choose_model == 'T5':
#                 time.sleep(1)
#                 output.text('T5 model has been chosen')


# Keyphrase_.subheader('Kindly type your Keyphrase here')

# with Keyphrase_.form(key="third", clear_on_submit=False):
#     text_keyphrase = Keyphrase_.text_area('Please input your text there:')
#     Keyphrase_.session_state.text=text_keyphrase