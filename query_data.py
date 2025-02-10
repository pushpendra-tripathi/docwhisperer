"""RAG Query Engine for document question answering.

This module implements the core RAG (Retrieval Augmented Generation) functionality
for querying documents using vector similarity search and LLM-based response generation.
"""

import argparse
import logging
import re
from functools import lru_cache
from typing import List, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama

from config import config
from get_embedding_function import get_embedding_function

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
CACHE_SIZE = 100

PROMPT_TEMPLATE = """
Answer the question based strictly on the following context:

{context}

---

Question: {question}

Instructions:
1. Analysis Phase:
   - wrap your analysis (thinking) in <think> </think> tags
   - First verify if the question is answerable using the context
   - Identify and quote relevant passages verbatim from the context
   - Perform step-by-step reasoning to connect evidence to the answer
   - Acknowledge and resolve any conflicting information from the context
   - State any assumptions or limitations in your analysis

2. Final Answer Requirements:
   [STRICT ENFORCEMENT MECHANISMS]
   3. Content Standards:
      - RAW EVIDENCE PRESENTATION REQUIRED:
        * Exhaustive presentation of relevant context excerpts
        * NO synthesis/interpretation - literal transcription only
        * Maintain original context sequence and relationships
        * Verbatim evidence chains must connect all claims

      - CONTEXT FIDELITY MANDATE:
        * Response structure MUST mirror context structure
        * NO reordering/grouping of related information
        * NO conceptual clustering - preserve document flow
        * Verbatim priority over paraphrasing (80/20 ratio)

      - ANTI-ABSTRACTION PROTOCOLS:
        * NO summarization under any circumstances
        * NO conceptual generalization
        * NO thematic grouping
        * ALL key terms/phrases must use direct quotations

   4. Structural Requirements:
      - EVIDENCE APPENDIX MANDATORY:
        * List ALL used verbatim quotes with exact sources
        * Map each claim to 2+ supporting quotations
        * Include chunk IDs and document origins

      - LITERAL TRANSCRIPTION RULES:
        * Preserve original context formatting/markup
        * Maintain numerical precision (no rounding)
        * Retain all qualifiers/adverbs from source
        * NO information density reduction

3. Response Format:
<think>
[Step-by-step analysis showing logical reasoning process]
</think>

[Final answer using bullet points or detailed paragraph]

4. Quality Controls:
   - Cross-validate using triple-quote verification:
     1. First occurrence in context
     2. Direct supporting evidence
     3. Corroborating secondary source
   - Implement military-grade precision requirements:
     * 100% claim-to-context traceability
     * Zero-tolerance for unsupported assertions
   - Enforce explicit sourcing:
     * (Document:Source.pdf|Chunk:3-14) style citations
     * Line-number references where available

Remember: Your credibility depends on accurate, context-based responses.
"""


class RAGQueryEngine:
    """RAG Query Engine for document question answering."""

    def __init__(self):
        """Initialize the RAG Query Engine with vector store and LLM."""
        try:
            self.db = Chroma(
                persist_directory=config.CHROMA_PATH,
                embedding_function=get_embedding_function(),
            )
            self.llm = Ollama(
                model=config.DEFAULT_MODEL,
                temperature=config.TEMPERATURE,
            )
            self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            logger.info("RAG Query Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Query Engine: {str(e)}")
            raise

    @lru_cache(maxsize=CACHE_SIZE)
    def _get_relevant_documents(self, question: str) -> List[Document]:
        """Get relevant documents for a question using similarity search.

        Args:
            question: The question to find relevant documents for

        Returns:
            List of relevant documents

        Raises:
            Exception: If document retrieval fails
        """
        try:
            return self.db.similarity_search(question, k=config.TOP_K_RESULTS)
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {str(e)}")
            raise

    def _format_documents(self, docs: List[Document]) -> str:
        """Format documents into a single context string, ensuring it doesn't exceed token limits."""
        context = "\n\n".join(doc.page_content for doc in docs)

        # Estimate token count (roughly 1 token = 4 characters)
        if len(context) > 2048 * 4:
            # If context is too large, take only the most relevant parts
            context = context[: 2048 * 4]
            logger.warning("Context was truncated to fit token limits")

        return context

    def _get_llm_response(
        self, context: str, question: str, retries: int = MAX_RETRIES
    ) -> str:
        """Get LLM response for a question and context.

        Args:
            context: The context to use for answering
            question: The question to answer
            retries: Number of retries on failure

        Returns:
            LLM response string

        Raises:
            Exception: If LLM query fails after retries
        """
        last_error = None
        for attempt in range(retries):
            try:
                chain = self.prompt | self.llm
                return str(chain.invoke({"context": context, "question": question}))
            except Exception as e:
                last_error = e
                if attempt == retries - 1:
                    logger.error(f"LLM query failed after {retries} attempts: {str(e)}")
                    raise
                logger.warning(f"LLM query attempt {attempt + 1} failed: {str(e)}")

        # This line should never be reached due to the raise in the loop,
        # but we add it to satisfy the type checker
        raise last_error or Exception("LLM query failed")

    def _extract_thinking(self, response: str) -> str:
        """Extract thinking context from between <think> tags with better error handling."""
        try:
            match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if match:
                return match.group(1).strip()

            logger.warning(
                "No thinking block found in LLM response. Full response:\n%s", response
            )
            return "No analytical thinking block found in response."
        except Exception as e:
            logger.error("Error extracting thinking context: %s", str(e))
            return "Error retrieving analysis."

    def _extract_answer(self, response: str) -> str:
        """Extract final answer with improved error handling."""
        try:
            parts = response.split("</think>")
            if len(parts) > 1:
                return parts[-1].strip()

            logger.warning("Answer separator missing. Using full response as answer.")
            return response.strip()
        except Exception as e:
            logger.error("Error extracting answer: %s", str(e))
            return "Error retrieving final answer."

    def _get_sources(self, docs: List[Document]) -> List[str]:
        """Get formatted source information from documents.

        Args:
            docs: List of source documents

        Returns:
            List of formatted source strings
        """
        return [f"Document: {doc.metadata.get('id', 'Unknown')}" for doc in docs]

    def query(self, question: str) -> Tuple[str, List[str], str]:
        """Query the RAG system with a question.

        Args:
            question: The question to answer

        Returns:
            Tuple containing (answer, sources, thinking_context)

        Raises:
            Exception: If query processing fails
        """
        try:
            logger.info(f"Processing query: {question}")

            # Get relevant documents
            docs = self._get_relevant_documents(question)
            if not docs:
                logger.warning("No relevant documents found")
                return "No relevant information found in the database.", [], ""

            # Format context and get LLM response
            context = self._format_documents(docs)
            response = self._get_llm_response(context, question)

            # Extract components
            thinking = self._extract_thinking(response)
            answer = self._extract_answer(response)
            sources = self._get_sources(docs)

            logger.info("Query processed successfully")
            return answer, sources, thinking

        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise


def query_rag(query_text: str) -> str:
    """Main entry point for RAG queries.

    Args:
        query_text: The question to ask

    Returns:
        Formatted response string

    Raises:
        Exception: If query fails
    """
    try:
        engine = RAGQueryEngine()
        answer, sources, thinking = engine.query(query_text)

        formatted_response = (
            f"Sources: {sources}\n\n" f"Thinking: {thinking}\n\n" f"Answer: {answer}"
        )
        print(formatted_response)
        return answer

    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise


def main():
    """CLI entry point with improved error handling."""
    try:
        parser = argparse.ArgumentParser(
            description="Query the RAG system with natural language questions"
        )
        parser.add_argument(
            "query_text", type=str, help="The question to ask the RAG system"
        )
        args = parser.parse_args()
        query_rag(args.query_text)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
