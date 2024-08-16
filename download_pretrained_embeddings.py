import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import argparse

def download_and_save_embeddings(model_name, output_file):
    # Load the model and tokenizer
    print(f"Loading model and tokenizer for {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get the embedding layer
    print("Extracting embeddings...")
    embedding_layer = model.get_input_embeddings()
    
    # Convert to numpy array
    embedding_weights = embedding_layer.weight.detach().cpu().numpy()

    # Save to file
    print(f"Saving embeddings to {output_file}...")
    np.save(output_file, embedding_weights)
    
    print(f"Embeddings saved. Shape: {embedding_weights.shape}")
    print(f"Vocabulary size: {len(tokenizer)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save word embeddings from a Hugging Face model.")
    parser.add_argument("model_name", type=str, help="Name of the model on Hugging Face")
    parser.add_argument("output_file", type=str, help="Output file name (without .npy extension)")
    
    args = parser.parse_args()
    
    download_and_save_embeddings(args.model_name, f"{args.output_file}.npy")