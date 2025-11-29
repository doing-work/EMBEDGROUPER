# Grouper: Large-Scale Company Name Grouping

A production-ready CLI tool for grouping company names at million-record scale using modern NLP embeddings and FAISS approximate nearest neighbor search.

## Features

- **Scalable**: Handles 1M+ company names efficiently
- **Fast**: GPU auto-detection with CPU fallback
- **Accurate**: Uses state-of-the-art sentence transformer embeddings
- **Configurable**: Customizable similarity thresholds, models, and clustering methods
- **Production-ready**: Includes progress tracking, error handling, and detailed statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, you may need to install `faiss-gpu` separately:

```bash
pip install faiss-gpu
```

The tool will auto-detect GPU availability and use it when available.

## Usage

### Basic Usage

```bash
python -m grouper.main --input companies.csv --output grouped.csv
```

### Specify Column Name

If your CSV uses a different column name for company names:

```bash
python -m grouper.main --input companies.csv --output grouped.csv --column "name"
```

### Custom Similarity Threshold

Adjust the similarity threshold (0.0 to 1.0) for clustering:

```bash
python -m grouper.main --input companies.csv --output grouped.csv --threshold 0.90
```

### Use Different Embedding Model

Use a multilingual model or other sentence transformer:

```bash
python -m grouper.main --input companies.csv --output grouped.csv --model paraphrase-multilingual-mpnet-base-v2
```

### Large Dataset Optimization

For very large datasets (1M+ records):

```bash
python -m grouper.main \
  --input large.csv \
  --output grouped.csv \
  --batch-size 64 \
  --top-k 100 \
  --index-type hnsw
```

### All Options

```bash
python -m grouper.main \
  --input companies.csv \
  --output grouped.csv \
  --column company_name \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --threshold 0.85 \
  --top-k 50 \
  --batch-size 32 \
  --index-type auto \
  --clustering connected_components \
  --canonical-method longest \
  --quiet
```

## Input Format

Input should be a CSV file with at least one column containing company names. By default, the tool looks for a column named `company_name`.

Example `companies.csv`:
```csv
company_name
Apple Inc.
Apple Inc
Apple Corporation
Microsoft Corp
Microsoft Corporation
Google LLC
Google Inc.
```

## Output Format

The output CSV contains the following columns:

- `original_name`: The original company name from input
- `cluster_id`: Unique identifier for the cluster this company belongs to
- `canonical_name`: The selected canonical name for this cluster
- `similarity_score_to_canonical`: Cosine similarity score to the canonical name (0.0 to 1.0)
- `neighbor_count`: Number of similar companies found
- `cluster_size`: Total number of companies in this cluster

Example output:
```csv
original_name,cluster_id,canonical_name,similarity_score_to_canonical,neighbor_count,cluster_size
Apple Inc.,0,Apple Inc.,1.0,2,3
Apple Inc,0,Apple Inc.,0.98,2,3
Apple Corporation,0,Apple Inc.,0.95,2,3
Microsoft Corp,1,Microsoft Corporation,0.99,1,2
Microsoft Corporation,1,Microsoft Corporation,1.0,1,2
Google LLC,2,Google LLC,1.0,1,2
Google Inc.,2,Google LLC,0.97,1,2
```

## Performance

### Estimated Runtime for 1 Million Company Names

| Stage                        | Time Estimate                                |
| ---------------------------- | -------------------------------------------- |
| Embedding generation         | 3–8 minutes (GPU) or 15–25 minutes (CPU)    |
| FAISS index training + build | 1–3 minutes                                  |
| ANN search                   | 2–6 minutes                                  |
| Clustering                   | 1–4 minutes                                  |
| **Total**                    | **5–20 minutes (GPU)** or **20–40 minutes (CPU)** |

### Optimization Tips

1. **Use GPU**: Install `faiss-gpu` for 3-5x speedup
2. **Increase batch size**: For large datasets, try `--batch-size 64` or `128`
3. **Adjust top-k**: Lower `--top-k` (e.g., 30) for faster processing if you have high similarity threshold
4. **Use HNSW index**: For datasets > 2M, use `--index-type hnsw`

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file path (required) | - |
| `--output` | Output CSV file path (required) | - |
| `--column` | Column name containing company names | `company_name` |
| `--model` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `--threshold` | Similarity threshold for clustering | `0.85` |
| `--top-k` | Number of neighbors to retrieve | `50` |
| `--batch-size` | Batch size for embedding generation | `32` |
| `--index-type` | FAISS index type (`auto`, `flat`, `hnsw`, `ivf`) | `auto` |
| `--clustering` | Clustering method | `connected_components` |
| `--canonical-method` | Method to select canonical name (`longest`, `most_frequent`, `centroid`) | `longest` |
| `--quiet` | Suppress progress messages | False |

## How It Works

1. **Normalization**: Company names are normalized (lowercase, remove legal suffixes, expand abbreviations)
2. **Embedding**: Each normalized name is converted to a dense vector using sentence transformers
3. **Indexing**: FAISS index is built for efficient similarity search
4. **Matching**: For each company, find top-k most similar companies using approximate nearest neighbor search
5. **Clustering**: Build a similarity graph and use connected components to group related companies
6. **Canonical Selection**: Select a representative canonical name for each cluster

## Technical Details

### Text Normalization

- Converts to lowercase
- Removes punctuation and accents
- Removes legal suffixes (inc, llc, ltd, corp, co, plc, gmbh, etc.)
- Removes filler words ("the", "company")
- Expands abbreviations (intl → international)

### Embedding Models

Default model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, fast and accurate)

Alternative models:
- `paraphrase-multilingual-mpnet-base-v2` (multilingual support)
- Any compatible sentence transformer model from Hugging Face

### FAISS Index Types

- **auto**: Automatically selects based on dataset size
- **flat**: Exact search, best for < 2M records
- **hnsw**: Hierarchical Navigable Small World, best for large datasets
- **ivf**: Inverted File Index, for very large datasets

### Clustering

Uses connected components on a similarity graph. Companies with similarity ≥ threshold are connected, and connected components form clusters.

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (e.g., `--batch-size 16`)
- Use `--index-type hnsw` for large datasets
- Process in chunks if dataset is extremely large

### Slow Processing

- Install `faiss-gpu` for GPU acceleration
- Increase `--batch-size` if you have sufficient memory
- Use a smaller embedding model
- Reduce `--top-k` if threshold is high

### Low Quality Clusters

- Lower `--threshold` to find more matches (e.g., `0.80`)
- Increase `--top-k` to search more neighbors
- Try a different embedding model
- Use `--canonical-method centroid` for better canonical selection

## License

This project is provided as-is for large-scale company name grouping tasks.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests.

