import streamlit as st
import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# App title and description
st.title("📚 RAG Chatbot with LangChain")
st.markdown("""
This application demonstrates a Retrieval-Augmented Generation (RAG) chatbot 
built using LangChain. Upload your dataset and start asking questions!
""")

class StreamlitRAGChatbot:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """Initialize the RAG Chatbot for Streamlit"""
        self.api_key = api_key
        self.model_name = model_name
        self.vectorstore = None
        self.rag_chain = None
        
    def load_data_from_file(self, uploaded_file):
        """Load data from an uploaded file"""
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # Read different file formats
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}. Please upload a CSV, JSON, or Excel file.")
                return None
                
            st.success(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns.")
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    def prepare_documents(self, df):
        """Convert DataFrame to documents and split into chunks"""
        # Check if the DataFrame is empty
        if df is None or df.empty:
            st.error("No data to process.")
            return None
            
        # Combine text columns if there are multiple
        if 'text' not in df.columns:
            # Create a text column by combining all string columns
            string_columns = df.select_dtypes(include=['object']).columns.tolist()
            df['text'] = df[string_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            
        # Load documents from DataFrame
        with st.spinner("Processing documents..."):
            loader = DataFrameLoader(df, page_content_column="text")
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            st.info(f"Split data into {len(splits)} chunks")
            return splits
    
    def create_vectorstore(self, splits):
        """Create a vector store from the document chunks"""
        if splits is None or len(splits) == 0:
            st.error("No document chunks to process.")
            return None
            
        # Create embeddings and vectorstore
        with st.spinner("Creating vector store (this may take a moment)..."):
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                vectorstore = Chroma.from_documents(
                    documents=splits, 
                    embedding=embeddings
                )
                
                self.vectorstore = vectorstore
                st.success("Vector store created successfully!")
                return self.vectorstore
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None
    
    def setup_rag_pipeline(self):
        """Set up the RAG pipeline using LangChain"""
        if self.vectorstore is None:
            st.error("Vector store not initialized. Please load data first.")
            return None
            
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create template for the prompt
        template = """You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question using only the provided context. If you cannot answer based on the context, say "I don't have enough information to answer this question." Be concise but comprehensive.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Set up the RAG chain
        try:
            llm = ChatOpenAI(
                model_name=self.model_name, 
                temperature=0,
                openai_api_key=self.api_key
            )
            
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            self.rag_chain = rag_chain
            return rag_chain
            
        except Exception as e:
            st.error(f"Error setting up RAG pipeline: {e}")
            return None
    
    def get_answer(self, question):
        """Get an answer for a user question"""
        if self.rag_chain is None:
            st.error("RAG pipeline not initialized. Please set up the chatbot first.")
            return "System error: Chatbot not properly initialized."
            
        try:
            with st.spinner("Thinking..."):
                response = self.rag_chain.invoke(question)
                return response
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Create sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input - first check for environment variable, then ask user
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API Key:", type="password", 
                           help="Your API key will not be stored after the session ends")
    else:
        st.success("API key loaded from environment variable!")
    
    # Model selection
    model_name = st.selectbox(
        "Select GPT Model:",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV, JSON, Excel):", 
                                   type=['csv', 'json', 'xlsx', 'xls'])
    
    # Set up chatbot button
    if st.button("Initialize Chatbot"):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not uploaded_file:
            st.error("Please upload a dataset file.")
        else:
            # Initialize chatbot with the provided API key
            st.session_state.chatbot = StreamlitRAGChatbot(api_key=api_key, model_name=model_name)
            
            # Load and process the data
            df = st.session_state.chatbot.load_data_from_file(uploaded_file)
            
            if df is not None:
                splits = st.session_state.chatbot.prepare_documents(df)
                
                if splits:
                    vectorstore = st.session_state.chatbot.create_vectorstore(splits)
                    
                    if vectorstore:
                        rag_chain = st.session_state.chatbot.setup_rag_pipeline()
                        
                        if rag_chain:
                            st.session_state.data_loaded = True
                            st.success("Chatbot is ready! You can start asking questions in the main panel.")
    
    # Load example dataset
    if st.button("Load Example Books Dataset"):
        # Create example data
        example_data = """title,author,genre,year,description
"To Kill a Mockingbird","Harper Lee","Fiction",1960,"The story of young Scout Finch and her father Atticus, a lawyer who defends a Black man accused of a terrible crime in a small Alabama town. A classic exploration of race, justice, and growing up in the American South during the 1930s."
"1984","George Orwell","Dystopian Fiction",1949,"Set in a totalitarian society ruled by Big Brother and the Party, this novel follows Winston Smith as he rebels against the oppressive regime. Known for introducing concepts like thoughtcrime, doublethink, and the Thought Police."
"The Great Gatsby","F. Scott Fitzgerald","Fiction",1925,"Set in the Jazz Age on Long Island, the novel depicts the mysterious millionaire Jay Gatsby and his obsession with the beautiful Daisy Buchanan. A critique of the American Dream and the excess of the Roaring Twenties."
"Pride and Prejudice","Jane Austen","Romance",1813,"The story follows Elizabeth Bennet as she deals with issues of manners, upbringing, morality, and marriage in the society of the landed gentry of the British Regency. Famous for the relationship between Elizabeth and Mr. Darcy."
"The Hobbit","J.R.R. Tolkien","Fantasy",1937,"Bilbo Baggins, a hobbit who enjoys a comfortable life, is swept into an epic quest to reclaim the lost Dwarf Kingdom of Erebor from the fearsome dragon Smaug. The prelude to The Lord of the Rings."
"Harry Potter and the Philosopher's Stone","J.K. Rowling","Fantasy",1997,"The first novel in the Harry Potter series, it follows Harry Potter, a young wizard who discovers his magical heritage on his eleventh birthday, when he receives a letter of acceptance to Hogwarts School of Witchcraft and Wizardry."
"The Catcher in the Rye","J.D. Salinger","Fiction",1951,"The story of Holden Caulfield, a teenage boy who has been expelled from prep school and wanders New York City. Known for its themes of teenage angst, alienation, and rebellion against adult society."
"The Lord of the Rings","J.R.R. Tolkien","Fantasy",1954,"An epic high-fantasy novel that follows hobbit Frodo Baggins as he and the Fellowship embark on a quest to destroy the One Ring, to ensure the destruction of its maker, the Dark Lord Sauron."
"Brave New World","Aldous Huxley","Dystopian Fiction",1932,"Set in a futuristic World State, where citizens are engineered and conditioned to serve society. The novel anticipates developments in reproductive technology, sleep-learning, and psychological manipulation."
"The Alchemist","Paulo Coelho","Fiction",1988,"A philosophical novel about a young Andalusian shepherd named Santiago who travels to Egypt after having a recurring dream of finding treasure there. An international bestseller about following one's dreams."
"""
        
        # Create a dataframe from the example data
        import io
        df = pd.read_csv(io.StringIO(example_data))
        
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            # Initialize chatbot with the provided API key
            st.session_state.chatbot = StreamlitRAGChatbot(api_key=api_key, model_name=model_name)
            
            # Process the dataframe directly
            if df is not None:
                splits = st.session_state.chatbot.prepare_documents(df)
                
                if splits:
                    vectorstore = st.session_state.chatbot.create_vectorstore(splits)
                    
                    if vectorstore:
                        rag_chain = st.session_state.chatbot.setup_rag_pipeline()
                        
                        if rag_chain:
                            st.session_state.data_loaded = True
                            st.success("Example dataset loaded! You can start asking questions in the main panel.")
    
    # Add a separator
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    This app demonstrates a RAG-based chatbot built with LangChain.
    
    Features:
    - Upload your own data (CSV, JSON, Excel)
    - Ask questions about your data
    - Get AI-generated responses based on your data
    
    Built with Streamlit and LangChain.
    """)

# Main chat interface
if st.session_state.data_loaded:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.get_answer(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    # Show instructions when chatbot is not initialized
    st.info("👈 Please enter your OpenAI API key and upload a dataset in the sidebar to get started.")
    
    # Display example questions
    st.markdown("### Example Questions You Can Ask:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - Tell me about [specific item in your dataset]
        - What are the main themes in [category]?
        - Summarize the information about [topic]
        - When was [item] created/published?
        """)
    
    with col2:
        st.markdown("""
        - Compare [item A] and [item B]
        - What's unique about [specific entry]?
        - Which items have [specific characteristic]?
        - What's the oldest/newest item in the dataset?
        """)
    
    # Add sample dataset description
    st.markdown("### Sample Dataset")
    st.markdown("""
    If you don't have a dataset ready, you can click "Load Example Books Dataset" in the sidebar.
    
    The example dataset contains information about 10 popular books including:
    - Book titles
    - Authors
    - Genres
    - Publication years
    - Book descriptions
    """)
