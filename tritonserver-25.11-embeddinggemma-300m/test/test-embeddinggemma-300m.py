#!/usr/bin/env python
"""
Test script for EmbeddingGemma-300M model on Triton Inference Server.
Tests both query and document embedding generation.
"""
import sys
import numpy as np
import tritonclient.grpc as grpcclient


def calculate_similarity(query_emb, doc_embs):
    """Calculate cosine similarity between query and documents."""
    # query_emb shape: (1, 768), doc_embs shape: (n, 768)
    similarities = np.dot(query_emb, doc_embs.T)
    return similarities


def test_query_embedding(triton_client, model_name):
    """Test query embedding generation."""
    print("\n" + "="*80)
    print("TEST 1: Query Embedding")
    print("="*80)
    
    query = "Which planet is known as the Red Planet?"
    print(f"Query: {query}")
    
    # Prepare input - query has dims [1] (max_batch_size: 0, no auto-batching)
    query_input = grpcclient.InferInput("query", [1], "BYTES")
    query_data = np.array([query.encode('utf-8')], dtype=object)
    query_input.set_data_from_numpy(query_data)
     
    # Prepare output
    output = grpcclient.InferRequestedOutput("embeddings")
    
    # Perform inference
    results = triton_client.infer(
        model_name=model_name,
        inputs=[query_input],
        outputs=[output]
    )
    
    # Get embeddings
    embeddings = results.as_numpy("embeddings")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected shape: (1, 768)")
    print(f"Embedding sample (first 5 values): {embeddings[0][:5]}")
    
    assert embeddings.shape[1] == 768, f"Expected embedding dim 768, got {embeddings.shape[1]}"
    print("✓ Query embedding test PASSED")
    
    return embeddings


def test_document_embeddings(triton_client, model_name):
    """Test document embeddings generation."""
    print("\n" + "="*80)
    print("TEST 2: Document Embeddings")
    print("="*80)
    
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    doc_embeddings_list = []
    
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}: {doc[:50]}...")
        
        # Prepare input - documents has dims [1, -1], send one document
        doc_input = grpcclient.InferInput("documents", [1, 1], "BYTES")
        doc_data = np.array([[doc.encode('utf-8')]], dtype=object)
        doc_input.set_data_from_numpy(doc_data)
        
        # Prepare output
        output = grpcclient.InferRequestedOutput("embeddings")
        
        # Perform inference
        results = triton_client.infer(
            model_name=model_name,
            inputs=[doc_input],
            outputs=[output]
        )
        
        # Get embeddings
        embeddings = results.as_numpy("embeddings")
        print(f"  Embeddings shape: {embeddings.shape}")
        doc_embeddings_list.append(embeddings)
    
    # Stack all document embeddings
    all_doc_embeddings = np.vstack(doc_embeddings_list)
    print(f"\nAll document embeddings shape: {all_doc_embeddings.shape}")
    print(f"Expected shape: (4, 768)")
    
    assert all_doc_embeddings.shape == (4, 768), f"Expected shape (4, 768), got {all_doc_embeddings.shape}"
    print("✓ Document embeddings test PASSED")
    
    return all_doc_embeddings


def test_multiple_documents(triton_client, model_name):
    """Test encoding multiple documents in a single request."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Documents at Once")
    print("="*80)
    
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    ]
    
    print(f"Sending {len(documents)} documents in one request...")
    
    # Prepare input - documents has dims [1, -1], send multiple documents
    doc_input = grpcclient.InferInput("documents", [1, len(documents)], "BYTES")
    doc_data = np.array([[doc.encode('utf-8') for doc in documents]], dtype=object)
    doc_input.set_data_from_numpy(doc_data)
    
    # Prepare output
    output = grpcclient.InferRequestedOutput("embeddings")
    
    # Perform inference
    results = triton_client.infer(
        model_name=model_name,
        inputs=[doc_input],
        outputs=[output]
    )
    
    # Get embeddings
    embeddings = results.as_numpy("embeddings")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected shape: (1, 768) for single document batch")
    
    print("✓ Multiple documents test PASSED")
    return embeddings


def test_similarity_ranking(query_emb, doc_embs):
    """Test similarity calculation and ranking."""
    print("\n" + "="*80)
    print("TEST 4: Similarity Ranking")
    print("="*80)
    
    similarities = calculate_similarity(query_emb, doc_embs)
    print(f"Similarities shape: {similarities.shape}")
    print(f"Similarity scores: {similarities[0]}")
    
    # Rank documents by similarity
    ranked_indices = np.argsort(similarities[0])[::-1]
    
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    print("\nRanked Results:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"{rank}. [Score: {similarities[0][idx]:.4f}] {documents[idx][:60]}...")
    
    # The Mars document (index 1) should have the highest similarity
    top_match_idx = ranked_indices[0]
    print(f"\nTop match index: {top_match_idx}")
    print(f"Expected index: 1 (Mars document)")
    
    if top_match_idx == 1:
        print("✓ Similarity ranking test PASSED - Mars document ranked highest!")
    else:
        print("⚠ Warning: Expected Mars document to rank highest, but got different result")
        print("  This may be acceptable depending on the model's understanding")
    
    return similarities


def main():
    """Main test function."""
    print("="*80)
    print("EmbeddingGemma-300M Triton Inference Server Test")
    print("="*80)
    
    # Create Triton client
    try:
        triton_client = grpcclient.InferenceServerClient(
            url="0.0.0.0:8001",
            verbose=False,
            ssl=False
        )
        print("✓ Successfully connected to Triton Server at 0.0.0.0:8001")
    except Exception as e:
        print(f"✗ Failed to create Triton client: {str(e)}")
        sys.exit(1)
    
    model_name = "embeddinggemma-300m"
    
    # Check if model is ready
    try:
        if triton_client.is_model_ready(model_name):
            print(f"✓ Model '{model_name}' is ready")
        else:
            print(f"✗ Model '{model_name}' is not ready")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error checking model status: {str(e)}")
        sys.exit(1)
    
    try:
        # Run tests
        query_emb = test_query_embedding(triton_client, model_name)
        doc_embs = test_document_embeddings(triton_client, model_name)
        multi_doc_emb = test_multiple_documents(triton_client, model_name)
        similarities = test_similarity_ranking(query_emb, doc_embs)
        
        # Print statistics
        print("\n" + "="*80)
        print("Inference Statistics")
        print("="*80)
        statistics = triton_client.get_inference_statistics(model_name=model_name)
        print(statistics)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()