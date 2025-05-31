import re
from typing import Tuple, List
from langchain_community.vectorstores import Chroma

class ResponseGenerator:
    def __init__(self, model, tokenizer):
        """
        Initialize the response generator.
        
        Args:
            model: The model to use for generation
            tokenizer: The tokenizer to use
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_response(self, text: str) -> str:
        """
        Generate a response for the given text.
        
        Args:
            text (str): Input text to generate response for
            
        Returns:
            str: Generated response
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        res = self.model.generate(
            tokens.to("cuda"),
            max_new_tokens=250,
            num_return_sequences=1,
            temperature=0.01,
            num_beams=1,
            top_p=0.95,
            top_k=50,
            do_sample=True
        ).to('cpu')
        
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)[0]

class RAGChatbot:
    PROMPT_TEMPLATE = """
    You are an AI assistant that answers a user query based on the provided context. Use only the relevant information from the retrieved context to generate the response.
    
    Carefully **combine relevant information** from all QA pairs and **synthesize a well-structured summary** without repetition.

    Context:
    {context}

    Now, answer the following question based on **retrieved context only**:

    User Query:
    {question}

    Answer:
    """

    def __init__(self, model, tokenizer):
        """
        Initialize the RAG chatbot.
        
        Args:
            model: The model to use for generation
            tokenizer: The tokenizer to use
        """
        self.response_generator = ResponseGenerator(model, tokenizer)

    def process_query(self, user_query: str, vec_db: Chroma) -> Tuple[str, List]:
        """
        Process a user query using RAG.
        
        Args:
            user_query (str): User's query
            vec_db (Chroma): Vector database instance
            
        Returns:
            Tuple[str, List]: Generated response and retrieved context
        """
        # Query the vector database
        retrieved_context = vec_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.60, "k": 3}
        ).invoke(user_query)

        if not retrieved_context:
            return "I don't know. I couldn't find the relevant information in the provided context.", []

        # Process retrieved context
        text = "\n".join(item.page_content for item in retrieved_context)
        answers = re.findall(r'A:\s*(.*?)(?=\s*Q:|$)', text, re.DOTALL)
        context_text = " ".join(answer.strip() for answer in answers)

        # Generate response
        prompt = self.PROMPT_TEMPLATE.format(context=text, question=user_query)
        response = self.response_generator.generate_response(prompt)

        return response, retrieved_context