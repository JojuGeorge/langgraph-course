from dotenv import load_dotenv
load_dotenv()

from pprint import pprint
from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocument = retrieval_grader.invoke({"question": question, "document": doc_txt})
    
    assert res.binary_score == "yes"
    
    
def test_retrieval_grader_answer_no() -> None:
    question = "Best pizza place?"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    
    res: GradeDocument = retrieval_grader.invoke({"question": question, "document": doc_txt})
    
    assert res.binary_score == "no"
    
    
def test_generation_chain() -> None: 
    question = "agent memory"
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)