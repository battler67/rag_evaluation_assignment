import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model {embedding_model_name} for evaluation...")
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """Computes cosine similarity between two texts using the evaluation embedding model."""
        embeddings = self.embedding_model.encode([text1, text2])
        v1, v2 = embeddings[0], embeddings[1]
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(similarity)

    def keyword_overlap(self, generated_answer: str, expected_answer: str) -> float:
        """Simple keyword overlap metric ignoring stop words and punctuation."""
        import string
        def tokenize(text):
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            words = text.split()
            # extremely basic stop word removal
            stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "of", "for", "with", "on", "it", "they", "that", "this"}
            return set([w for w in words if w not in stopwords])

        expected_tokens = tokenize(expected_answer)
        generated_tokens = tokenize(generated_answer)
        
        if not expected_tokens:
            return 0.0
            
        overlap = expected_tokens.intersection(generated_tokens)
        return len(overlap) / len(expected_tokens)

    def evaluate_qa_pair(self, question: str, generated_answer: str, expected_answer: str, retrieved_contexts: List[str] = None) -> Dict[str, Any]:
        """Evaluates a single generated answer against the expected answer."""
        similarity = self.compute_cosine_similarity(generated_answer, expected_answer)
        overlap = self.keyword_overlap(generated_answer, expected_answer)
        
        result = {
            "question": question,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "quantitative_metrics": {
                "cosine_similarity": similarity,
                "keyword_overlap": overlap
            },
            "qualitative_assessment": {
                "coherence_score": 0, # To be filled by human
                "factual_correctness_score": 0, # To be filled by human
                "completeness_score": 0, # To be filled by human
                "notes": ""
            }
        }
        
        # Optional Retrieval Evaluation
        if retrieved_contexts:
            # Check if expected answer's keywords are present in retrieved contexts
            expected_tokens = set([w.lower() for w in expected_answer.split() if len(w) > 3])
            context_text = " ".join(retrieved_contexts).lower()
            found_tokens = [t for t in expected_tokens if t in context_text]
            retrieval_score = len(found_tokens) / len(expected_tokens) if expected_tokens else 0
            
            result["retrieval_performance"] = {
                "keyword_presence_in_context": retrieval_score
            }
            
        return result

    def evaluate_all(self, qa_dataset_path: str, output_path: str, rag_app) -> List[Dict]:
        """Runs evaluation over a dataset and saves results."""
        with open(qa_dataset_path, 'r') as f:
            qa_data = json.load(f)
            
        results = []
        for item in qa_data:
            question = item['question']
            expected = item['expected_answer']
            
            logger.info(f"Evaluating Question: {question}")
            
            try:
                response = rag_app.query(question)
                generated = response['answer']
                contexts = [doc.page_content for doc in response['source_documents']]
                
                eval_result = self.evaluate_qa_pair(question, generated, expected, contexts)
            except Exception as e:
                logger.error(f"Error evaluating {question}: {e}")
                eval_result = {
                    "question": question,
                    "error": str(e)
                }
                
            results.append(eval_result)
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Evaluation complete. Results saved to {output_path}")
        return results
