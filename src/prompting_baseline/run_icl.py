import os
import json
import tqdm
import time
import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI

SCRATCH_PSEUDOCODE_SYNTAX = """Scratch pseudocode syntax rules and requirements:
1. Output only Scratch pseudocode, with one block per line and no markdown.
2. Use exact Scratch-style block text from the examples. Do not output opcode keys.
3. Stack/command blocks are plain lines, e.g. move (10) steps.
4. Reporter inputs must be wrapped in parentheses, e.g. (x position), ((score) + (1)).
5. Boolean conditions must be wrapped in angle brackets, e.g. <mouse down?>, <touching (edge v)?>.
6. Text/name inputs use square brackets, e.g. [message1], [costume1], [score v].
7. Menu/dropdown inputs include the v marker, e.g. (space v), (random position v), [all v].
8. Control blocks use exact forms such as repeat (10), forever, if <condition> then, if <condition> then else, wait until <condition>.
9. Indent nested blocks with exactly 4 spaces and close every C-block with end.
10. For if else, use: if <condition> then, true branch, else, false branch, end.
11. Preserve names, messages, numbers, signs, decimal values, and action order from the natural language.
12. If no event is described, do not invent one."""

BATCH_ENDPOINT = "/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_REASONING_EFFORT = "auto"
DEFAULT_TEMPERATURE = None
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TRAIN_FILE = SCRIPT_DIR / "Updated_Dataset_May_NL2Scratch/splits/train.jsonl"
DEFAULT_DIAGNOSTIC_800_FILE = (
    REPO_ROOT
    / "dataset_analysis/sac_primary_composite_secondary_subset/sac_primary_composite_secondary_800.jsonl"
)
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "output/predictions_gpt54_diagnostic_800.jsonl"


def load_jsonl_data(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_predictions(predictions, file_path):
    """Save predictions to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"Saved predictions to {file_path}")


def pseudocode_to_string(pseudocode):
    """Convert pseudocode list to string."""
    if isinstance(pseudocode, list):
        return '\n'.join(pseudocode)
    return str(pseudocode)


def tokenize(text):
    """Small dependency-free tokenizer for TF-IDF retrieval."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def build_tfidf_index(train_data):
    """Build a lightweight TF-IDF index using only the Python standard library."""
    doc_tokens = [tokenize(item["nl"]) for item in train_data]
    doc_freq = defaultdict(int)
    for tokens in doc_tokens:
        for token in set(tokens):
            doc_freq[token] += 1

    n_docs = max(1, len(doc_tokens))
    idf = {
        token: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }

    vectors = []
    for tokens in doc_tokens:
        counts = Counter(tokens)
        weighted = {token: count * idf[token] for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        vectors.append({token: value / norm for token, value in weighted.items()})

    return {"idf": idf, "vectors": vectors}


def vectorize_query(query_nl, idf):
    counts = Counter(tokenize(query_nl))
    weighted = {
        token: count * idf[token]
        for token, count in counts.items()
        if token in idf
    }
    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    return {token: value / norm for token, value in weighted.items()}


def search_similar_examples(query_nl, train_data, tfidf_index, nshot=5):
    """Return top-k training examples by TF-IDF cosine similarity."""
    if nshot <= 0:
        return []

    query_vec = vectorize_query(query_nl, tfidf_index["idf"])
    if not query_vec:
        return train_data[:nshot]

    scores = []
    for idx, doc_vec in enumerate(tfidf_index["vectors"]):
        score = sum(query_weight * doc_vec.get(token, 0.0) for token, query_weight in query_vec.items())
        scores.append((score, idx))

    top_indexes = [idx for _, idx in sorted(scores, reverse=True)[:nshot]]
    return [train_data[i] for i in top_indexes]


def build_prompt(query_nl, examples, include_pseudocode=True):
    """
    Build the prompt for GPT with few-shot examples.
    """
    prompt = "Task: Convert natural language descriptions into Scratch pseudocode.\n\n"
    prompt += SCRATCH_PSEUDOCODE_SYNTAX + "\n\n"
    
    # Add few-shot examples
    if include_pseudocode:
        prompt += "Here are some examples:\n\n"
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Natural Language: {example['nl']}\n"
            prompt += f"Pseudocode:\n{pseudocode_to_string(example['pseudocode'])}\n\n"
    
    # Add the query
    prompt += "Now, please generate the Scratch pseudocode for the following description:\n\n"
    prompt += f"Natural Language: {query_nl}\n"
    prompt += "Pseudocode:\n"
    
    return prompt


def resolve_reasoning_effort(model, reasoning_effort):
    if reasoning_effort in ("", "none", None):
        return None
    if reasoning_effort == "auto":
        return "medium" if model.startswith("gpt-5") else None
    return reasoning_effort


def make_chat_body(model, prompt, max_tokens=DEFAULT_MAX_TOKENS,
                   reasoning_effort=DEFAULT_REASONING_EFFORT,
                   temperature=DEFAULT_TEMPERATURE):
    """Build the Chat Completions request body used by sync and batch modes."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    resolved_reasoning_effort = resolve_reasoning_effort(model, reasoning_effort)
    if resolved_reasoning_effort:
        body["reasoning_effort"] = resolved_reasoning_effort
    return body


def extract_chat_content(body):
    """Extract generated text from a Chat Completions response body."""
    return body["choices"][0]["message"]["content"].strip()


def call_gpt_api(client, model, prompt, max_tokens=DEFAULT_MAX_TOKENS,
                 reasoning_effort=DEFAULT_REASONING_EFFORT,
                 temperature=DEFAULT_TEMPERATURE, max_retries=3):
    """Call OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                **make_chat_body(model, prompt, max_tokens, reasoning_effort, temperature)
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(20)  # Wait before retry
            else:
                return f"ERROR: {str(e)}"


def output_related_path(output_file, suffix):
    path = Path(output_file)
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def build_icl_requests(train_file, test_file, model=DEFAULT_MODEL, nshot=5,
                       max_samples=None, max_tokens=DEFAULT_MAX_TOKENS,
                       reasoning_effort=DEFAULT_REASONING_EFFORT,
                       temperature=DEFAULT_TEMPERATURE):
    """Build batch requests plus metadata needed to reconstruct prediction rows."""
    print(f"Loading retrieval examples from training data: {train_file}")
    train_data = load_jsonl_data(train_file)
    print(f"Loaded {len(train_data)} training examples")
    print("Building TF-IDF retrieval index...")
    tfidf_index = build_tfidf_index(train_data)

    print(f"Loading evaluation inputs from: {test_file}")
    test_data = load_jsonl_data(test_file)

    if max_samples:
        test_data = test_data[:max_samples]
    print(f"Preparing {len(test_data)} batch requests with {model}...")

    requests = []
    metadata_rows = []
    for i, test_item in enumerate(tqdm.tqdm(test_data, desc="Preparing batch")):
        query_nl = test_item['nl']
        gold_pseudocode = test_item['pseudocode']
        similar_examples = search_similar_examples(query_nl, train_data, tfidf_index, nshot=nshot)
        prompt = build_prompt(query_nl, similar_examples)
        custom_id = f"nl2scratch-{i}"

        requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": make_chat_body(model, prompt, max_tokens, reasoning_effort, temperature),
        })
        metadata_rows.append({
            "custom_id": custom_id,
            "order": i,
            "key": test_item["key"],
            "nl": query_nl,
            "gold_pseudocode": gold_pseudocode,
            "similar_examples_keys": [ex["key"] for ex in similar_examples],
        })

    return requests, metadata_rows


def write_jsonl(rows, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def submit_batch(train_file, test_file, output_file, model=DEFAULT_MODEL, nshot=5,
                 max_samples=None, max_tokens=DEFAULT_MAX_TOKENS,
                 reasoning_effort=DEFAULT_REASONING_EFFORT,
                 temperature=DEFAULT_TEMPERATURE):
    """Create a Batch API input file, upload it, and submit the batch."""
    client = OpenAI()
    batch_input_file = output_related_path(output_file, "_batch_input")
    metadata_file = output_related_path(output_file, "_batch_metadata")

    requests, metadata_rows = build_icl_requests(
        train_file=train_file,
        test_file=test_file,
        model=model,
        nshot=nshot,
        max_samples=max_samples,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    write_jsonl(requests, batch_input_file)
    write_jsonl(metadata_rows, metadata_file)

    print(f"Uploading batch input: {batch_input_file}")
    with batch_input_file.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata={
            "task": "nl2scratch_icl",
            "model": model,
            "nshot": str(nshot),
            "reasoning_effort": reasoning_effort or "default",
            "temperature": str(temperature) if temperature is not None else "default",
        },
    )

    print("Batch submitted.")
    print(f"batch_id: {batch.id}")
    print(f"status: {batch.status}")
    print(f"input_file: {batch_input_file}")
    print(f"metadata_file: {metadata_file}")
    print(f"Collect later with: python nl2scratch_icl.py --mode batch_collect --batch_id {batch.id} --output_file {output_file}")
    return batch.id


def wait_for_batch(batch_id, poll_interval=60, wait_timeout=None):
    """Poll a batch until it reaches a terminal state."""
    client = OpenAI()
    start = time.time()
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"Batch {batch_id} status: {batch.status}")
        if batch.status in terminal_statuses:
            return batch
        if wait_timeout is not None and time.time() - start >= wait_timeout:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}; last status: {batch.status}")
        time.sleep(poll_interval)


def collect_batch(batch_id, output_file, metadata_file=None):
    """Download a completed batch output and convert it to prediction JSONL."""
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch_id} status: {batch.status}")

    if batch.status != "completed":
        print("Batch is not completed yet. Try collecting again later.")
        return

    if not batch.output_file_id:
        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            error_output_file = output_related_path(output_file, "_batch_errors")
            content = client.files.content(error_file_id).read()
            error_output_file.parent.mkdir(parents=True, exist_ok=True)
            error_output_file.write_bytes(content)
            print(f"Batch completed without output_file_id. Error file saved to {error_output_file}")
            try:
                first_error = load_jsonl_data(error_output_file)[0]
                print("First batch error:")
                print(json.dumps(first_error, ensure_ascii=False, indent=2))
            except Exception:
                pass
            raise RuntimeError(f"Batch {batch_id} completed with errors; see {error_output_file}")
        raise RuntimeError(f"Batch {batch_id} completed without output_file_id or error_file_id.")

    metadata_path = Path(metadata_file) if metadata_file else output_related_path(output_file, "_batch_metadata")
    metadata = {
        row["custom_id"]: row
        for row in load_jsonl_data(metadata_path)
    }

    raw_output_file = output_related_path(output_file, "_batch_raw_output")
    content = client.files.content(batch.output_file_id).read()
    raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_output_file.write_bytes(content)

    predictions = []
    batch_rows = load_jsonl_data(raw_output_file)
    for row in batch_rows:
        custom_id = row["custom_id"]
        base = metadata[custom_id]
        error = row.get("error")
        response = row.get("response") or {}
        if error:
            generated_pseudocode = f"ERROR: {error}"
        elif response.get("status_code") != 200:
            generated_pseudocode = f"ERROR: status {response.get('status_code')} {response.get('body')}"
        else:
            generated_pseudocode = extract_chat_content(response["body"])

        predictions.append({
            "order": base["order"],
            "key": base["key"],
            "nl": base["nl"],
            "gold_pseudocode": base["gold_pseudocode"],
            "predicted_pseudocode": generated_pseudocode,
            "similar_examples_keys": base["similar_examples_keys"],
        })

    predictions.sort(key=lambda r: r["order"])
    for prediction in predictions:
        del prediction["order"]
    save_predictions(predictions, output_file)
    print(f"Raw batch output saved to {raw_output_file}")


def run_batch_and_collect(train_file, test_file, output_file, model=DEFAULT_MODEL, nshot=5,
                          max_samples=None, max_tokens=DEFAULT_MAX_TOKENS,
                          reasoning_effort=DEFAULT_REASONING_EFFORT,
                          temperature=DEFAULT_TEMPERATURE,
                          poll_interval=60, wait_timeout=None):
    """Submit a batch, wait for completion, then collect predictions."""
    batch_id = submit_batch(
        train_file=train_file,
        test_file=test_file,
        output_file=output_file,
        model=model,
        nshot=nshot,
        max_samples=max_samples,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
    batch = wait_for_batch(batch_id, poll_interval=poll_interval, wait_timeout=wait_timeout)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")
    collect_batch(batch_id=batch_id, output_file=output_file)


def run_icl_experiment(train_file, test_file, output_file, model=DEFAULT_MODEL,
                       nshot=5, save_interval=50, max_samples=None,
                       max_tokens=DEFAULT_MAX_TOKENS,
                       reasoning_effort=DEFAULT_REASONING_EFFORT,
                       temperature=DEFAULT_TEMPERATURE):
    """
    Run in-context learning experiment on NL2Scratch task.
    """
    print(f"Loading retrieval examples from training data: {train_file}")
    train_data = load_jsonl_data(train_file)
    print(f"Loaded {len(train_data)} training examples")
    print("Building TF-IDF retrieval index...")
    tfidf_index = build_tfidf_index(train_data)
    
    print(f"Loading evaluation inputs from: {test_file}")
    test_data = load_jsonl_data(test_file)
    
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"Testing on {len(test_data)} examples")
    
    predictions = []
    
    client = OpenAI()

    print(f"\nRunning {nshot}-shot ICL with {model} synchronously...\n")
    pbar = tqdm.tqdm(test_data, desc="Generating predictions")
    
    for i, test_item in enumerate(pbar):
        query_nl = test_item['nl']
        gold_pseudocode = test_item['pseudocode']
        
        # Find similar examples from training set
        similar_examples = search_similar_examples(query_nl, train_data, tfidf_index, nshot=nshot)
        
        # Build prompt
        prompt = build_prompt(query_nl, similar_examples)
        
        # Call GPT API
        generated_pseudocode = call_gpt_api(
            client,
            model,
            prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )
        
        # Store prediction
        prediction = {
            'key': test_item['key'],
            'nl': query_nl,
            'gold_pseudocode': gold_pseudocode,
            'predicted_pseudocode': generated_pseudocode,
            'similar_examples_keys': [ex['key'] for ex in similar_examples]
        }
        predictions.append(prediction)
        
        # Print sample output
        if i < 3 or i % 20 == 0:
            print(f"\n{'='*80}")
            print(f"Sample {i+1}:")
            print(f"NL: {query_nl}")
            print(f"\nGold:\n{pseudocode_to_string(gold_pseudocode)}")
            print(f"\nPredicted:\n{generated_pseudocode}")
            print(f"{'='*80}\n")
        
        # Save intermediate results
        if (i + 1) % save_interval == 0:
            intermediate_file = output_file.replace('.jsonl', f'_checkpoint_{i+1}.jsonl')
            save_predictions(predictions, intermediate_file)
    
    # Save final results
    save_predictions(predictions, output_file)
    print(f"\nExperiment completed! Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='NL2Scratch In-Context Learning with GPT')
    parser.add_argument('--mode', type=str, default='batch_submit',
                        choices=['batch_submit', 'batch_collect', 'batch_run', 'sync'],
                        help='Use batch_run for one-command submit/wait/collect, batch_submit for async submit only, batch_collect to fetch results, or sync for small debugging runs')
    parser.add_argument('--train_file', type=str, 
                        default=str(DEFAULT_TRAIN_FILE),
                        help='Path to training data used to retrieve few-shot examples (JSONL)')
    parser.add_argument('--test_file', type=str,
                        default=str(DEFAULT_DIAGNOSTIC_800_FILE),
                        help='Path to evaluation input data (defaults to diagnostic 800 JSONL)')
    parser.add_argument('--output_file', type=str,
                        default=str(DEFAULT_OUTPUT_FILE),
                        help='Path to save predictions')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='OpenAI model to use')
    parser.add_argument('--batch_id', type=str, default=None,
                        help='Batch ID to collect when --mode batch_collect')
    parser.add_argument('--metadata_file', type=str, default=None,
                        help='Metadata JSONL from batch_submit; defaults to output_file with _batch_metadata suffix')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help='Maximum output tokens per request')
    parser.add_argument('--reasoning_effort', type=str, default=DEFAULT_REASONING_EFFORT,
                        choices=['auto', '', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh'],
                        help='Reasoning effort for GPT models; auto uses medium for gpt-5 models and omits it otherwise')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help='Optional temperature. Omit by default because gpt-5.4 only supports the default value.')
    parser.add_argument('--nshot', type=int, default=5,
                        help='Number of few-shot examples')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N samples')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of test samples (for testing)')
    parser.add_argument('--poll_interval', type=int, default=60,
                        help='Seconds between Batch API status checks in batch_run mode')
    parser.add_argument('--wait_timeout', type=int, default=None,
                        help='Optional max seconds to wait in batch_run mode')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else 'output', 
                exist_ok=True)
    
    if args.mode == 'batch_submit':
        submit_batch(
            train_file=args.train_file,
            test_file=args.test_file,
            output_file=args.output_file,
            model=args.model,
            nshot=args.nshot,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
        )
    elif args.mode == 'batch_run':
        run_batch_and_collect(
            train_file=args.train_file,
            test_file=args.test_file,
            output_file=args.output_file,
            model=args.model,
            nshot=args.nshot,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            poll_interval=args.poll_interval,
            wait_timeout=args.wait_timeout,
        )
    elif args.mode == 'batch_collect':
        if not args.batch_id:
            raise ValueError("--batch_id is required when --mode batch_collect")
        collect_batch(
            batch_id=args.batch_id,
            output_file=args.output_file,
            metadata_file=args.metadata_file,
        )
    else:
        run_icl_experiment(
            train_file=args.train_file,
            test_file=args.test_file,
            output_file=args.output_file,
            model=args.model,
            nshot=args.nshot,
            save_interval=args.save_interval,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
