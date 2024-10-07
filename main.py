import streamlit as st
import os
from openai import OpenAI
import base64
import json
from urllib.parse import urlparse
# import fitz
import openai
from PIL import Image
from io import BytesIO
from llama_index.core import VectorStoreIndex,Document,SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

save_dir = "uploaded_pdf"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
st.title("Fintech Chatbot")
st.write("Please upload pdf files  before asking any question other wise you cannot input your questions")
st.sidebar.title("Upload pdf files")
uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
Api_key = st.text_input("Please enter Openai  api key")
file_saved = False
openaiinit = False

if Api_key:
 client = OpenAI(api_key=Api_key) #Best practice needs OPENAI_API_KEY environment variable
 openai.api_key = Api_key
 openaiinit = True


def encode_image(image_path):
    with open(image_path,"rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pdf_to_images(pdf_path,output_dir):
    pdf_document = fitz.open(pdf_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image = Image.open(BytesIO(pix.tobytes()))
        image_file_name = f"page_{page_num+1}.png"
        image_file_path = os.path.join(output_dir,image_file_name)
        image.save(image_file_path,"PNG")


if uploaded_files:
    if os.path.exists(save_dir):
      
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
  
    for uploaded_file in uploaded_files:
        save_path = os.path.join(save_dir,uploaded_file.name)
        with open(save_path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File '{uploaded_file.name}' saved sucessfully")
    file_saved = True

file_imagesaved = "outputuserpdfimages"
text = ""
if file_saved and openaiinit:
    st.write("Give us some time to  preprocess documents")
    pdf_to_images(save_path,file_imagesaved)
    # for filename in os.listdir(file_imagesaved):
    #     word = ""
    #     file_path = os.path.join(file_imagesaved,filename)
    #     print(file_path)
    #     if os.path.isfile(file_path):
    #       base64_img = f"data:image/png;base64,{encode_image(file_path)}"
    #     response = client.chat.completions.create(
    #         model='gpt-4o',
    #        messages=[
    #       {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "What’s in this image with transactional value with numeric data  and payment date and contact information of user and bank who is issueing credit card and tell me breifly about payment method and how charges being calculated  and also tell about late payment,other charges,cash advance,extende credit  and charges in detail ."},
    #             {
    #                 "type": "image_url",
    #                 # for online images
    #                 # "image_url": {"url": image_url}
    #                 "image_url": {"url": f"{base64_img}"}
    #              }
    #          ],
    #      }
    #      ],
    #      )
    #     word = response.choices[0].message.content
    #     print(word)
    #     text += word
        
    # with open('data.txt','w',encoding='utf-8') as file:
    #     file.write(text)

    BANK_QA_AGENT_PERSONA = """
                          You are an experienced bank customer service and financial advisor with 10 years of experience in SBI. Your expertise includes:
                         - In-depth knowledge of banking products such as loans, credit cards, savings accounts, and investments.
                         - Strong understanding of banking regulations, compliance, and customer protection laws.
                         - STRONG understanding of credit card intrest rates of diffrent bank and should know how to resolve query regarding credit card.
                         - Experience in resolving customer issues, handling queries regarding transactions, fees,credit cards  and account management.
                         - Proficient in advising customers on financial products, managing risk, and optimizing their financial health.
                         - Ability to provide clear, concise, and accurate information while ensuring customer satisfaction.
                         Provide insights, analysis, and recommendations based on your expertise. Use relevant banking terminology and explain your reasoning clearly.
                       """
    custom_query_prompt = PromptTemplate(
    """
    {bank_qa_agent_persona}

    Context information is below.
    ---------------------
    {context_str}
    ---------------------

    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}

    Your response should include:
    1. A concise overview of the query (account type, transaction, loan,credit cards etc.)
    2. Key factors or regulations to consider while paying credit card bills
    3. Relevant banking product or service details.
    4.Please summarize the transaction history for the customer like transactions,amount paid,any pending payment or overdue charges.
    5. Please handles the transactions related to credit or debit card.

    6. Potential implications for the customer’s financial situation
    7. Any caveats or limitations based on account terms or regulations

    Answer:
    """
)
    custom_refine_prompt = PromptTemplate(
    """
    {bank_qa_agent_persona}

    We have provided an existing answer: {existing_answer}

    We have the opportunity to refine the existing answer with some more context below.
    ------------
    {context_msg}
    ------------

    Given the new context, refine the original answer to better address the query: {query_str}. 
    If the context isn't useful, return the original answer.

    Refined Answer:
    """
   )
    
  
    documents=SimpleDirectoryReader("data1").load_data()
    pc = Pinecone(api_key="264040b3-b298-4918-9d56-b31134d5ba48")
    pinecone_index = pc.Index("bankagent")
    vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
)   
    Settings.chunk_size = 512

    settings = Settings.embed_model = OpenAIEmbedding()
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store,settings=settings)
    query_engine = RetrieverQueryEngine.from_args(
    retriever=index.as_retriever(similarity_top_k=10),
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    text_qa_template=custom_query_prompt,
    refine_template=custom_refine_prompt,
    llm=OpenAI(model="gpt-4", max_tokens=600, temperature=0.0),
   )
    query_engine_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="Bank_QA_aGENT",  
        description="Helpful for resolving client query"
    ),
)   
    agent = OpenAIAgent.from_tools([query_engine_tool], verbose=True)

    print(index)
    query_engine = index.as_query_engine()
    st.write("File uploaded sucessfully.Now you can ask Questions")
    user_question = st.text_input("Your input")
    print(user_question)
    if user_question:
     response = agent.chat(user_question)
     print(response)
     st.write(response.response)
   



