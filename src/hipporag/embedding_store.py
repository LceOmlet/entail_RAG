import os

from src.hipporag.llm import _get_llm_class
# Disable Ray auto-initialization
os.environ["RAY_DISABLE_INIT"] = "1"

import asyncio
import numpy as np
from tqdm import tqdm
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput, text_processing, run_with_dynamic_pool

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}
    
    def insert_entities(self, entity_nodes2chunk_ids: Dict[str, List[str]], entity_nodes2triples: Dict[str, List[List[str]]], chunk_to_rows: Dict[str, Dict],  same_entity_threshold: float, prompt_template_manager, global_config):
        definitions = []
        async def disambiguate_entity(entity, passages, same_entity_threshold, prompt_template_manager, llm_model, embeddings, entities_with_definition=None, definition_embeddings=None):
            
            if entities_with_definition is None:
                entities_with_definition = []

            if len(passages) == 0:
                return entities_with_definition
            if len(passages + entities_with_definition) == 1:
                entities_with_definition.append(entity)
                return entities_with_definition
            
            # take account of the current definitions
            num_current_definitions = len(entities_with_definition)
            
            # get the first paragraph to define the entity
            entity_definition = prompt_template_manager.render(name="disambiguation_paragraph", named_entity=entity, passage=passages[0])
            entity_definition, meta_data = llm_model.infer_(entity_definition)
            
            if entity_definition in ["date.", "number.", "time."]:
                return [f"{entity}@def: {entity_definition}"]

            this_definition_embedding = self.embedding_model.batch_encode([f"{entity}: {entity_definition}"])

            embeddings =  np.concatenate([embeddings[1:]]) 
            if definition_embeddings is not None:
                embeddings = np.concatenate([embeddings, definition_embeddings]) 
            
            # the definition entails the same entity
            similarities = this_definition_embedding[0] @ embeddings.T
            connections = similarities > same_entity_threshold

            # print(len(connections), len(passages), len(entities_with_definition), len(embeddings))

            assert len(connections) == len(passages) + num_current_definitions - 1    

            collapsed = False
            if num_current_definitions > 0:
                cd_connections = connections[-num_current_definitions:]
                connections = connections[:len(passages) - 1]
                first_true_idx = None
                indices_to_remove = []
                
                # First pass: find the first True and mark other True indices for removal
                for i in range(len(cd_connections)):
                    if cd_connections[i]:
                        if first_true_idx is None:
                            # Keep the first True connection
                            first_true_idx = i
                            collapsed = True
                        else:
                            # Mark this index for removal
                            indices_to_remove.append(i)
                
                # Second pass: remove marked indices in reverse order
                for idx in sorted(indices_to_remove, reverse=True):
                    entities_with_definition.pop(idx)
                    if definition_embeddings is not None:
                        # Remove the corresponding embedding
                        definition_embeddings = np.concatenate([
                            definition_embeddings[:idx],
                            definition_embeddings[idx+1:]
                        ])
            if not collapsed:
                entities_with_definition.append(f"{entity}@def: {entity_definition}")
                if definition_embeddings is not None:
                    definition_embeddings = np.concatenate([definition_embeddings, this_definition_embedding])
                else:
                    definition_embeddings = this_definition_embedding
            
            # Remove passages that are too similar to the definition (index 0)
            filtered_passages = []  # Keep the definition
            # connections = connections[1:]
            embeddings_ = []
            for i in range(len(connections)):
                if not connections[i]:  # If similar to definition
                    filtered_passages.append(passages[i+1])  # i+1 because connections is shorter by 1
                    embeddings_.append(embeddings[i])
            passages = filtered_passages
            embeddings = np.array(embeddings_)

            # print(len(passages), len(embeddings), len(entities_with_definition), len(definition_embeddings))

            # other entities should be disambiguated
            return await disambiguate_entity(entity, passages, same_entity_threshold, prompt_template_manager, llm_model, embeddings, entities_with_definition, definition_embeddings)
        
        async def process_entity(entity):
            passages = [f"A textual record about \"{entity}\": {chunk_to_rows[chunk_id]['content']}" for chunk_id in entity_nodes2chunk_ids[entity]]
            embeddings = self.embedding_model.batch_encode(passages)
            return await disambiguate_entity(entity, passages, same_entity_threshold, prompt_template_manager, _get_llm_class(global_config), embeddings)

        async def process_all_entities():
            # Use the generic dynamic pool function
            entities = list(entity_nodes2chunk_ids.keys())
            
            def on_result_success(entities_with_definition):
                definitions.extend(entities_with_definition)
                print(entities_with_definition)
            
            await run_with_dynamic_pool(
                items=entities,
                process_func=process_entity,
                max_concurrent_tasks=200,
                show_progress=True,
                description="Processing entities",
                on_success=on_result_success,
                use_thread_pool=True
            )
        
        # Run the entity processing
        asyncio.run(process_all_entities())
        self.insert_strings(definitions)

    def union_find(self, connections: np.ndarray) -> List[Set[int]]:
        """
        Find connected components in the adjacency matrix using Union-Find algorithm.
        
        Args:
            connections: A 2D numpy array representing the adjacency matrix where connections[i][j] = True
                        means there is an edge between node i and node j.
        
        Returns:
            A list of sets, where each set contains the indices of nodes in a connected component.
        """
        n = len(connections)
        parent = list(range(n))  # Initialize parent array
        
        def find(x: int) -> int:
            """Find the root of node x with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int):
            """Union two sets containing x and y."""
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        
        # Union all connected nodes
        for i in range(n):
            for j in range(i + 1, n):
                if connections[i][j]:
                    union(i, j)
        
        # Group nodes by their root
        components: Dict[int, Set[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)
        
        return list(components.values())

    def insert_strings(self, texts: List[str]):
        nodes_dict = {}

        for text in texts:
            nodes_dict[compute_mdhash_id(text.split("@def:")[0], prefix=self.namespace + "-")] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        logger.info(
            f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def delete(self, hash_ids):
        indices = []

        for hash in hash_ids:
            indices.append(self.hash_id_to_idx[hash])

        sorted_indices = np.sort(indices)[::-1]

        for idx in sorted_indices:
            self.hash_ids.pop(idx)
            self.texts.pop(idx)
            self.embeddings.pop(idx)

        logger.info(f"Saving record after deletion.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]

    def get_hash_id(self, text):
        return self.text_to_hash_id[text]

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings