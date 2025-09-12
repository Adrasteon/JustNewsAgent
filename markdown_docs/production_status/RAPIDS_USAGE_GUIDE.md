---
title: RAPIDS Integration Guide
description: Auto-generated description for RAPIDS Integration Guide
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# RAPIDS Integration Guide

## Overview

This guide provides comprehensive information about using RAPIDS 25.04 in the JustNewsAgent project. RAPIDS is a suite of open-source libraries that provide GPU-accelerated data science and machine learning capabilities.

## Environment Setup

### Primary Environment
```bash
conda activate justnews-v2-py312
```

This environment includes:
- Python 3.12.11
- RAPIDS 25.04
- CUDA 12.4
- PyTorch 2.5.1+cu124

### Alternative Environment
```bash
conda activate justnews-v2-prod  # Python 3.11 environment
```

## RAPIDS Libraries Available

### Core Libraries

#### cudf - GPU DataFrames
```python
import cudf

# Read CSV files directly to GPU
df = cudf.read_csv('news_articles.csv')

# Perform pandas-compatible operations on GPU
df['sentiment_score'] = df['text'].str.lower()
grouped = df.groupby('category').agg({'sentiment_score': 'mean'})

# Convert back to pandas if needed
pandas_df = df.to_pandas()
```

#### cuml - GPU Machine Learning
```python
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.cluster import KMeans

# Classification
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# Clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data)
```

#### cugraph - GPU Graph Analytics
```python
import cugraph

# Create graph from edge list
G = cugraph.Graph()
G.from_cudf_edgelist(df, source='source_node', destination='target_node')

# Calculate PageRank
pagerank = cugraph.pagerank(G)

# Find shortest paths
distances = cugraph.shortest_path(G, source=0)
```

#### cuspatial - GPU Spatial Analytics
```python
import cuspatial

# Point-in-polygon operations
points = cuspatial.GeoSeries.from_points(lon, lat)
polygons = cuspatial.GeoSeries.from_polygons(...)
result = cuspatial.point_in_polygon(points, polygons)

# Distance calculations
distances = cuspatial.haversine_distance(lon1, lat1, lon2, lat2)
```

#### cuvs - GPU Vector Search
```python
import cuvs

# Build vector index
index = cuvs.Index(cuvs.IndexType.IVF_PQ, metric=cuvs.DistanceType.L2)
index.build(vectors)

# Search for nearest neighbors
distances, indices = index.search(query_vectors, k=10)
```

## Performance Optimization

### Memory Management
```python
import cudf

# Set memory allocator
cudf.set_allocator("managed")  # or "default"

# Monitor memory usage
print(f"GPU Memory Used: {cudf.get_memory_usage()}")

# Clear GPU memory
import gc
gc.collect()
cudf.clear_cache()
```

### Data Transfer Optimization
```python
# Minimize CPU-GPU transfers
# Process data entirely on GPU when possible

# Use cudf operations instead of pandas
gpu_df = cudf.DataFrame(data)  # Direct creation
result = gpu_df.operation()    # GPU computation

# Batch operations for efficiency
batch_size = 10000
for i in range(0, len(data), batch_size):
    batch = cudf.DataFrame(data[i:i+batch_size])
    # Process batch on GPU
```

## Integration with JustNewsAgent

### Data Processing Pipeline
```python
import cudf
import cuml
from cuml.feature_extraction.text import TfidfVectorizer

class RAPIDSDataProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def process_articles(self, articles_df):
        # Convert to GPU DataFrame
        gpu_df = cudf.DataFrame(articles_df)

        # Text preprocessing on GPU
        gpu_df['clean_text'] = gpu_df['content'].str.lower()

        # Feature extraction
        features = self.vectorizer.fit_transform(gpu_df['clean_text'])

        # Sentiment analysis with GPU ML
        sentiment_model = cuml.LogisticRegression()
        sentiment_model.fit(features, gpu_df['sentiment_labels'])

        return gpu_df, features
```

### News Analysis Example
```python
import cudf
import cugraph
import cuml

def analyze_news_network(articles_df):
    """Analyze relationships between news articles using GPU graph analytics"""

    # Create GPU DataFrame
    df = cudf.DataFrame(articles_df)

    # Build similarity graph
    # (Implementation would calculate article similarities)

    # Find communities using GPU
    from cuml.cluster import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=5)
    communities = clusterer.fit_predict(feature_matrix)

    # Graph analysis
    G = cugraph.Graph()
    # Add edges based on article relationships

    # Calculate centrality measures
    betweenness = cugraph.betweenness_centrality(G)

    return communities, betweenness
```

## Best Practices

### 1. Data Size Considerations
- RAPIDS excels with datasets > 1GB
- For smaller datasets, CPU processing may be faster due to transfer overhead
- Consider batching for large datasets

### 2. Memory Management
- Monitor GPU memory usage regularly
- Use `cudf.clear_cache()` to free memory
- Set appropriate batch sizes to avoid memory overflow

### 3. Error Handling
```python
try:
    # RAPIDS operations
    result = cudf_operation()
except cudf.errors.MemoryError:
    # Handle out of memory
    cudf.clear_cache()
    # Retry with smaller batch
except Exception as e:
    print(f"RAPIDS error: {e}")
    # Fallback to CPU processing
```

### 4. Performance Monitoring
```python
import time

def benchmark_operation(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Operation took {end - start:.2f} seconds")
    return result

# Usage
gpu_result = benchmark_operation(cudf_operation, data)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Version Mismatch
```
Error: CUDA version mismatch
```
**Solution**: Ensure RAPIDS and PyTorch use compatible CUDA versions
```bash
conda list | grep cuda
nvidia-smi  # Check GPU CUDA version
```

#### 2. Memory Errors
```
Error: Out of memory
```
**Solutions**:
- Reduce batch size
- Use `cudf.clear_cache()`
- Switch to CPU processing for smaller datasets

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'cudf'
```
**Solution**: Activate correct environment
```bash
conda activate justnews-v2-py312
python -c "import cudf; print('RAPIDS working')"
```

### Performance Tuning

#### GPU Utilization Check
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check RAPIDS GPU usage
python -c "import cudf; print(cudf.get_memory_usage())"
```

#### Optimization Tips
1. **Minimize data transfers** between CPU and GPU
2. **Use appropriate data types** (float32 vs float64)
3. **Batch operations** when possible
4. **Profile code** to identify bottlenecks
5. **Use GPU-specific algorithms** when available

## Migration from CPU Processing

### Before (CPU-only)
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')
model = RandomForestClassifier()
model.fit(df.drop('target', axis=1), df['target'])
```

### After (GPU-accelerated)
```python
import cudf
from cuml.ensemble import RandomForestClassifier

df = cudf.read_csv('data.csv')
model = RandomForestClassifier()
model.fit(df.drop('target', axis=1), df['target'])
```

## Resources

- [RAPIDS Documentation](https://docs.rapids.ai/)
- [cuDF API Reference](https://docs.rapids.ai/api/cudf/stable/)
- [cuML API Reference](https://docs.rapids.ai/api/cuml/stable/)
- [cuGraph API Reference](https://docs.rapids.ai/api/cugraph/stable/)
- [NVIDIA RAPIDS GitHub](https://github.com/rapidsai)

## Support

For RAPIDS-related issues:
1. Check the [RAPIDS issue tracker](https://github.com/rapidsai/cudf/issues)
2. Verify environment setup with the validation script
3. Review GPU memory usage and system requirements
4. Consider CPU fallback for smaller datasets

---

*Last Updated: August 31, 2025*
*Environment: justnews-v2-py312 (Python 3.12.11, RAPIDS 25.04)*</content>
<parameter name="filePath">/home/adra/JustNewsAgent/docs/RAPIDS_USAGE_GUIDE.md

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

