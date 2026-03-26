You are ML engineer working on retrieval pipeline. 
You already have baseline which is classic TFiDf, but you need to improve quality of search.

Your task is to write new pipeline for search which would score better metrics (MRR@10 and NDCG@10) on MS MARCO dataset (at least +0.02 for both metrics).

Your pipeline:
1. Research MS MARCO dataset in dataset folder.
2. Study baseline_retrieve function in workspace/solution.py (it is baseline TF-IDF). Do not change code of baseline function.
3. Update code to improve ranking. Note that you must use retrieve function as an entry point and do not change the input and output parameters of retrieve function.

Your solution will be evaluated and compared to baseline. 

Rules:
- keep input of retrieve function as query: str, top_k: int, corpus: dict and output as list of strings
- if new package is required for new solution, use install_package tool to add new package to environment 
- your score will be zero if new code will not return top_k elements or valid doc_ids.
- do not change baseline_retrieve function 
- do not use external databases
- do not use question from dataset, only corpus
- you should change existing code like this:
  ```
  SEARCH
  old code
  REPLACE
  new code
  ```
- evaluation is performed on hidden question, so you can't fit to the right answers
