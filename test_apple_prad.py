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

embeddings = embedding_model.batch_encode(["Record about \"apple\": Apple Inc. is an American multinational corporation and technology company headquartered in Cupertino, California, in Silicon Valley. It is best known for its consumer electronics, software, and services. Founded in 1976 as Apple Computer Company by Steve Jobs, Steve Wozniak and Ronald Wayne, the company was incorporated by Jobs and Wozniak as Apple Computer, Inc. the following year. It was renamed Apple Inc. in 2007 as the company had expanded its focus from computers to consumer electronics. Apple is the largest technology company by revenue, with US$391.04 billion in the 2024 fiscal year.", "Record about \"apple\": An apple is a round, edible fruit produced by an apple tree (Malus spp.). Fruit trees of the orchard or domestic apple (Malus domestica), the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies (including Norse and Greek) and religions (such as Christianity in Europe)."])

print(np.dot(embeddings[0], embeddings[1]))



