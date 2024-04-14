import streamlit as st
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from huggingface_hub import InferenceClient

output_directory = 'output_chunks'
directory_of_embeddings = 'output_chunks/embeddings'

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

from sklearn.metrics.pairwise import cosine_similarity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embedding_for_text(text,tokenizer = tokenizer,model = model):
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def generate_embedding_for_question(text,tokenizer = tokenizer,model = model):
    # Tokenize sentences
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

def cosine_similarity_score(embedding1, embedding2):
    return cosine_similarity(embedding1.cpu().numpy().reshape(1, -1), embedding2.cpu().numpy().reshape(1, -1))

def top_five_files(question, directory_of_embeddings):
  question_embedding = generate_embedding_for_question(question)
  file_similarity_scores = []
  for filename in os.listdir(directory_of_embeddings):
    if filename.endswith('.pt'):
      file_path = os.path.join(directory_of_embeddings, filename)
      file_embedding = torch.load(file_path).to(device)
      similarity_score = cosine_similarity_score(question_embedding, file_embedding)
      file_similarity_scores.append((filename, similarity_score))
  sorted_files = sorted(file_similarity_scores, key=lambda x: x[1], reverse=True)
  top_5_files = sorted_files[:5]
  return top_5_files

def get_context(question, directory_of_embeddings):
  # print('#########################################################################################')
  # print('question: ', question)
  final_context = ''
  top_files = top_five_files(question, directory_of_embeddings)
  # print(top_files)
  text_files = []
  for item in top_files:
    text_files.append(item[0][:3]+'.txt') # for top 5 file name
  # print(text_files)
  for text in os.listdir(output_directory):
    if text in text_files:
      # print(text)
      with open(os.path.join(output_directory, text) , 'r', encoding='utf-8') as file:
      # with open(os.path.join(output_directory, text) , 'r') as file:
        final_context += file.read()
  # print('directory_of_texts: ', directory_of_texts)
  # print('final_context: ', final_context)
  # print('#########################################################################################')
  return final_context, text_files

class QA:
  def __init__(self, question, answer, doc_id):
    self.question = question
    self.answer = answer
    self.doc_id = doc_id
    self.context = ''

def output(query):
  list_of_qs = [
                # QA('what is the name of the company?',
  #                 'US magazine Mobile PC','009.txt'),
                QA(query,
                   'Restaurants','152.txt')

              ]
  result = []
  top_5_matches = []
  def get_answer(question, context):
      client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token= "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")
      # HF_TOKEN = "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"
      # client = InferenceClient(model="meta-llama/Llama-2-7b-chat-hf", token=HF_TOKEN)
      res = client.text_generation(f'{context} answer the following question based on the above context {question}  note: only give the crisp answer in the format Answer : ', max_new_tokens=2300)
      return res
  for item in list_of_qs:
    final_context, top_matches = get_context(item.question, directory_of_embeddings)
    # print(final_context)
    # print(item.question)
    answer = get_answer(item.question, final_context)
    result.append(answer)

  return answer

st.title("Finance ChatBot")
def process_query(query):
    # Example logic to process the query
    return output(query.lower())

def main():

    # Accept user input
    query = st.text_input('Enter your query here this model till now is only trained on the data of alphabet inc. :')

    if st.button('Submit'):
        # Process the query
        result = process_query(query)
        
        # Display the result
        st.write('Response:')

        st.write(result.lower())

if __name__ == '__main__':
    main()
