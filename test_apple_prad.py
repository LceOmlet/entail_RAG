from src.hipporag.embedding_model import _get_embedding_model_class
from src.hipporag.utils.config_utils import BaseConfig
import numpy as np

global_config = BaseConfig(
    save_dir="outputs/apple_prad",
    llm_base_url="http://localhost:8000/v1",
    llm_name="microsoft/phi-4",
    dataset="apple_prad",
    embedding_model_name="facebook/contriever",
    force_index_from_scratch=True,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
    force_openie_from_scratch=True,
    rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
    retrieval_top_k=200,
    linking_top_k=5,
    max_qa_steps=3,
    qa_top_k=5,
    graph_type="facts_and_sim_passage_node_unidirectional",
    embedding_batch_size=8,
    max_new_tokens=None,
    corpus_len=1,
    openie_mode="online"
)

global_config.embedding_model_name = "facebook/contriever"
embedding_model = _get_embedding_model_class(
                embedding_model_name=global_config.embedding_model_name)(global_config=global_config,
                                                                              embedding_model_name=global_config.embedding_model_name)

embeddings = embedding_model.batch_encode(["lionel messi: a professional footballer, known for his time at FC Barcelona and his early development at La Masia.", "lionel messi: a footballer, recognized as La Liga's all-time top goalscorer and record holder for most goals in a season."])

print(np.dot(embeddings[0], embeddings[1]))



