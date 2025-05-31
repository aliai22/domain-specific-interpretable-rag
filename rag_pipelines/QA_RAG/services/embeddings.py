class EmbeddingGenerator:
    def __init__(self, model):
        """
        Initialize the embedding generator with a model.
        
        Args:
            model: The model to use for generating embeddings
        """
        self.model = model

    def generate_embeddings(self, text):
        """
        Generate embeddings for the input text.
        
        Args:
            text: Text or list of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(text)
        return embeddings.tolist()

class LocalEmbeddingFunction:
    def __init__(self, embedd_model):
        """
        Initialize the embedding function with a model.
        
        Args:
            embedd_model: The model to use for generating embeddings
        """
        self.embedding_generator = EmbeddingGenerator(embedd_model)

    def __call__(self, input):
        """
        Generate embeddings for input documents.
        
        Args:
            input: Input documents
            
        Returns:
            Generated embeddings
        """
        return self.embedding_generator.generate_embeddings(input)

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        return self.embedding_generator.generate_embeddings(texts)

    def embed_query(self, query):
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embedding_generator.generate_embeddings(query)