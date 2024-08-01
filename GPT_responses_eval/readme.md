1. **Generate Answers:**

   Run `answers.py` to generate answers using the LLM Llama3:8b. This script will generate answers for two scenarios:
   
   - **LLM only:** Using only the LLM without any retrieval.
   - **With retrieval:** Using the text corresponding to the first 1000 tokens from the concatenation of the retrieved chunks by the 5 retrieval methods: bm25, sim, mmr, mlq, and reo.

   ```sh
   python answers.py
   ```

2. **Evaluate Generated Answers:**

   Run `gpt_eval.py` to evaluate the generated answers using GPT-4. Ensure you have your OpenAI API key inserted in the script before running it.

   ```sh
   python gpt_eval.py
   ```

   *Make sure to insert your OpenAI API Key in the script.*

## Files Description

- `answers.py`: Script to generate answers using the LLM Llama3:8b for two scenarios: LLM only and with retrieval.
- `gpt_eval.py`: Script to evaluate the generated answers using GPT-4. Insert your OpenAI API key in this script.


