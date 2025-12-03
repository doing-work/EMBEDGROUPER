"""CLI entry point for company name grouping."""

import argparse
import sys

# Handle both module and direct execution
try:
    from .processor import CompanyGrouper
except ImportError:
    # Fallback for direct execution
    from processor import CompanyGrouper


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Group company names using embeddings and FAISS approximate nearest neighbor search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m grouper.main --input companies.csv --output grouped.csv
  
  # Specify column name
  python -m grouper.main --input companies.csv --output grouped.csv --column "name"
  
  # Custom threshold and model
  python -m grouper.main --input companies.csv --output grouped.csv --threshold 0.90 --model paraphrase-multilingual-mpnet-base-v2
  
  # Large dataset with custom batch size
  python -m grouper.main --input large.csv --output grouped.csv --batch-size 64 --top-k 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path containing company names'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path for grouped results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--column',
        type=str,
        default='company_name',
        help='Column name containing company names (default: company_name)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for clustering (default: 0.85)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Number of nearest neighbors to retrieve per company (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    
    parser.add_argument(
        '--index-type',
        type=str,
        default='auto',
        choices=['auto', 'flat', 'hnsw', 'ivf', 'ivfpq'],
        help='FAISS index type (default: auto). ivfpq uses Product Quantization for memory efficiency on very large datasets'
    )
    
    parser.add_argument(
        '--clustering',
        type=str,
        default='connected_components',
        choices=['connected_components', 'hdbscan', 'agglomerative', 'minhash', 'hybrid'],
        help='Clustering method (default: connected_components). '
             'minhash: Fast MinHash-based matching. '
             'hybrid: MinHash for exact matches + embeddings for semantic matching (recommended). '
             'hdbscan/agglomerative: Work directly on embeddings without FAISS search'
    )
    
    parser.add_argument(
        '--use-memmap',
        action='store_true',
        help='Use memory-mapped files for embeddings (recommended for large datasets to reduce RAM usage)'
    )
    
    parser.add_argument(
        '--cache-embeddings',
        action='store_true',
        default=True,
        help='Cache embeddings to disk for reuse (default: True, saves time on subsequent runs)'
    )
    
    parser.add_argument(
        '--no-cache-embeddings',
        dest='cache_embeddings',
        action='store_false',
        help='Disable embedding caching'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of embeddings even if cache exists'
    )
    
    parser.add_argument(
        '--reduce-dimensions',
        type=int,
        default=None,
        metavar='N',
        help='Reduce embedding dimensions using PCA (e.g., 256, 128, 64). Reduces memory usage and speeds up FAISS operations'
    )
    
    parser.add_argument(
        '--preserve-variance',
        type=float,
        default=0.95,
        help='Variance to preserve when using PCA (0.0-1.0, default: 0.95). Higher values preserve more information but may use more dimensions'
    )
    
    parser.add_argument(
        '--canonical-method',
        type=str,
        default='longest',
        choices=['longest', 'most_frequent', 'centroid'],
        help='Method to select canonical name per cluster (default: longest)'
    )
    
    parser.add_argument(
        '--max-cluster-size',
        type=int,
        default=1000,
        help='Maximum cluster size before splitting large clusters (default: 1000, set to 0 to disable splitting)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    
    # Initialize processor
    grouper = CompanyGrouper(
        model_name=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold,
        top_k=args.top_k,
        index_type=args.index_type,
        clustering_method=args.clustering,
        canonical_method=args.canonical_method,
        max_cluster_size=args.max_cluster_size,
        use_memmap=args.use_memmap,
        cache_embeddings=args.cache_embeddings,
        force_regenerate=args.force_regenerate,
        reduce_dimensions=args.reduce_dimensions,
        preserve_variance=args.preserve_variance,
        verbose=not args.quiet
    )
    
    # Process
    try:
        stats = grouper.process(
            input_file=args.input,
            output_file=args.output,
            column_name=args.column
        )
        
        print(f"Successfully processed {stats['total_records']:,} records")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

