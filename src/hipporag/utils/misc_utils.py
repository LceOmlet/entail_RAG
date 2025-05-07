from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )

async def run_with_dynamic_pool(
    items: list, 
    process_func, 
    max_concurrent_tasks: int = 10, 
    show_progress: bool = True, 
    description: str = "Processing",
    on_success=None,
    on_error=None,
    use_thread_pool: bool = True,
    verbose_logging: bool = False
):
    """
    Run a collection of items through an async processing function using a dynamic task pool.
    When tasks complete, new ones are automatically added to maintain concurrency.
    
    Args:
        items: List of items to process
        process_func: Async function that takes a single item and processes it
        max_concurrent_tasks: Maximum number of concurrent tasks to run
        show_progress: Whether to show a progress bar
        description: Description for the progress bar
        on_success: Optional callback function to run on successful completion of each task
        on_error: Optional callback function to run when a task raises an exception
        use_thread_pool: If True, run potentially blocking functions in a thread pool
        verbose_logging: Enable detailed logging of task execution status
    
    Returns:
        List of results from all successful tasks
    """
    import asyncio
    from tqdm import tqdm
    import logging
    import concurrent.futures
    import time
    import inspect
    
    logger = logging.getLogger(__name__)
    
    if verbose_logging:
        logger.info(f"Starting run_with_dynamic_pool with {len(items)} items, max_concurrent_tasks={max_concurrent_tasks}, use_thread_pool={use_thread_pool}")
    
    start_time = time.time()
    results = []
    
    progress_bar = tqdm(total=len(items), desc=description) if show_progress else None
    
    # Track tasks with item indexes for better debugging
    task_to_item = {}
    completed_count = 0
    error_count = 0
    
    # Track remaining items to process
    remaining_items = items.copy()
    
    # Check if process_func is actually an async function
    is_process_func_async = inspect.iscoroutinefunction(process_func)
    if verbose_logging:
        logger.info(f"Process function is{'n' if not is_process_func_async else ''} async: {is_process_func_async}")
    
    # Create thread pool for potentially blocking functions
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) if use_thread_pool else None
    if verbose_logging and executor:
        logger.info(f"Created ThreadPoolExecutor with {max_concurrent_tasks} workers")
    
    async def wrapped_process_func(item, item_index):
        """Wrap the process function to handle both async and sync functions"""
        func_start_time = time.time()
        
        if verbose_logging:
            logger.info(f"Starting task {item_index+1}/{len(items)} using {'thread pool' if use_thread_pool and executor else 'async call'}")
        
        try:
            if use_thread_pool and executor:
                if is_process_func_async:
                    # For async functions in thread pool, we need a helper
                    async def thread_helper():
                        loop = asyncio.get_running_loop()
                        def run_coroutine_in_thread():
                            # Create a new event loop in the thread
                            async_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(async_loop)
                            try:
                                # Run the coroutine and return its result
                                return async_loop.run_until_complete(process_func(item))
                            finally:
                                async_loop.close()
                        
                        # Run the helper in a thread
                        return await loop.run_in_executor(executor, run_coroutine_in_thread)
                    
                    result = await thread_helper()
                else:
                    # Simple case: run non-async function in thread pool
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(executor, process_func, item)
            else:
                # Run as normal async function
                if is_process_func_async:
                    result = await process_func(item)
                else:
                    # Handle non-async function in async context
                    result = process_func(item)
            
            # Check if result is a coroutine that wasn't awaited
            if inspect.iscoroutine(result):
                logger.warning(f"Task {item_index+1} returned a coroutine that wasn't awaited. Will await it now.")
                result = await result
            
            elapsed = time.time() - func_start_time
            if verbose_logging:
                logger.info(f"Task {item_index+1} completed successfully in {elapsed:.2f}s")
            
            return result
        except Exception as e:
            elapsed = time.time() - func_start_time
            logger.error(f"Task {item_index+1} failed after {elapsed:.2f}s with error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def add_tasks_to_pool():
        """Add new tasks to the pool up to max_concurrent_tasks"""
        added_count = 0
        
        while len(task_to_item) < max_concurrent_tasks and remaining_items:
            item_index = len(items) - len(remaining_items)
            item = remaining_items.pop(0)
            
            task = asyncio.create_task(wrapped_process_func(item, item_index))
            task_to_item[task] = (item, item_index)
            added_count += 1
        
        if verbose_logging and added_count > 0:
            logger.info(f"Added {added_count} new tasks to pool. Current pool size: {len(task_to_item)}, Remaining items: {len(remaining_items)}")
    
    try:
        # Initial filling of the task pool
        add_tasks_to_pool()
        
        # Process until all tasks are completed
        cycle_count = 0
        while task_to_item:
            cycle_count += 1
            cycle_start = time.time()
            
            if verbose_logging:
                logger.info(f"Cycle {cycle_count}: Waiting for tasks to complete. Active tasks: {len(task_to_item)}, Completed: {completed_count}, Errors: {error_count}, Remaining: {len(remaining_items)}")
            
            # Wait for at least one task to complete with timeout
            done, pending = await asyncio.wait(
                task_to_item.keys(), 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5.0  # Add timeout to avoid hanging indefinitely
            )
            
            cycle_elapsed = time.time() - cycle_start
            if verbose_logging:
                logger.info(f"Cycle {cycle_count} completed in {cycle_elapsed:.2f}s with {len(done)} tasks done")
            
            if not done:
                logger.warning(f"Cycle {cycle_count}: No tasks completed within 5s timeout. This may indicate blocking operations. Active tasks: {len(task_to_item)}")
                # Diagnostics on pending tasks
                if verbose_logging:
                    for i, task in enumerate(task_to_item.keys()):
                        item, idx = task_to_item[task]
                        logger.info(f"  Pending task {i+1}: item index {idx}, task done: {task.done()}, cancelled: {task.cancelled()}")
                continue
            
            # Process completed tasks
            for task in done:
                item, item_index = task_to_item.pop(task)
                
                try:
                    result = task.result()
                    
                    # Double-check if result is a coroutine (this shouldn't happen, but just in case)
                    if inspect.iscoroutine(result):
                        logger.warning(f"Task {item_index+1} result is a coroutine! This indicates an error in wrapped_process_func.")
                        logger.warning("Attempting to await the coroutine to get the actual result.")
                        result = await result
                    
                    results.append(result)
                    completed_count += 1
                    
                    if on_success:
                        if verbose_logging:
                            logger.info(f"Calling on_success callback for task {item_index+1}")
                        try:
                            on_success(result)
                        except Exception as callback_error:
                            logger.error(f"Error in on_success callback for task {item_index+1}: {callback_error}")
                            import traceback
                            traceback.print_exc()
                            # Continue processing other tasks rather than failing everything
                            error_count += 1
                except Exception as e:
                    error_count += 1
                    if on_error:
                        if verbose_logging:
                            logger.info(f"Calling on_error callback for task {item_index+1}")
                        on_error(e)
                    else:
                        logger.error(f"Error processing item {item_index+1}: {e}")
                        import traceback
                        traceback.print_exc()
                  
                if progress_bar:
                    progress_bar.update(1)
            
            # Immediately refill the task pool after processing completed tasks
            add_tasks_to_pool()
    
    finally:
        total_elapsed = time.time() - start_time
        logger.info(f"Finished processing: {completed_count} completed, {error_count} errors in {total_elapsed:.2f}s")
        
        if progress_bar:
            progress_bar.close()
        
        if executor:
            if verbose_logging:
                logger.info("Shutting down thread pool executor")
            executor.shutdown(wait=False)
    
    return results