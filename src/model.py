import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .config import Config

class LegalBertClassifier:
    def __init__(self, model_dir: str):
        print(f"Loading model from: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        # Map ids to labels
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        print(f"Model loaded with {len(self.id2label)} labels: {list(self.id2label.values())}")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

    def _chunk_by_tokens(self, text: str, max_tokens: int, stride: int) -> List[Dict[str, torch.Tensor]]:
        # Ensure max_tokens doesn't exceed model's position embedding limit
        max_model_tokens = getattr(self.model.config, 'max_position_embeddings', 512)
        if max_tokens > max_model_tokens:
            print(f"Warning: max_tokens ({max_tokens}) exceeds model limit ({max_model_tokens}), using {max_model_tokens}")
            max_tokens = max_model_tokens
        
        # Tokenize once; then window over input_ids with stride
        enc = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_ids = enc["input_ids"][0]
        attention_mask = torch.ones_like(input_ids)
       

        chunks = []
        # Ensure we don't exceed max_tokens for any chunk
        step = max_tokens - stride
        if step <= 0:
            print(f"Warning: step ({step}) is <= 0, using max_tokens as step")
            step = max_tokens
        print(f"Using step size: {step}")
        for start in range(0, len(input_ids), step):
            end = min(start + max_tokens, len(input_ids))
            ids_slice = input_ids[start:end]
            mask_slice = attention_mask[start:end]
            
            print(f"Chunk {len(chunks)+1}: start={start}, end={end}, length={len(ids_slice)}")
            
            # Ensure the chunk doesn't exceed max_tokens
            if len(ids_slice) > max_tokens:
                print(f"Truncating chunk from {len(ids_slice)} to {max_tokens} tokens")
                ids_slice = ids_slice[:max_tokens]
                mask_slice = mask_slice[:max_tokens]
            
            # Add special tokens per chunk with proper padding
            try:
                chunk = self.tokenizer.prepare_for_model(
                    ids_slice,
                    attention_mask=mask_slice,
                    truncation=True,
                    max_length=max_tokens,
                    padding="max_length",
                    return_tensors="pt"
                )
                chunks.append(chunk)
                print(f"Successfully created chunk {len(chunks)}")
            except Exception as e:
                print(f"Error creating chunk: {str(e)}")
                # Fallback: use simple tokenization
                chunk = self.tokenizer(
                    self.tokenizer.decode(ids_slice),
                    truncation=True,
                    max_length=max_tokens,
                    padding="max_length",
                    return_tensors="pt"
                )
                chunks.append(chunk)
                print(f"Created fallback chunk {len(chunks)}")
            if end == len(input_ids):
                break
        if not chunks:
            # Empty doc — build a minimal input
            chunks = [self.tokenizer("", return_tensors="pt", max_length=max_tokens, padding="max_length")]
        return chunks

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float], int]:
        """
        Returns: (top_label, top_confidence, probs_per_label, chunks_used)
        - Aggregates probabilities across chunks via mean.
        """
        # Short-circuit: tiny text
        if not text or not text.strip():
            return ("", 0.0, {}, 0)

        try:
            chunks = self._chunk_by_tokens(text, Config.MAX_TOKENS, Config.STRIDE_TOKENS)
            print(f"Number of chunks created: {len(chunks)}")
            if not chunks:
                print("No chunks created, returning empty result")
                return ("", 0.0, {}, 0)
                
            probs_sum = None
            for i, ch in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                ch = {k: v.to(self.device) for k, v in ch.items()}
                logits = self.model(**ch).logits
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
                print(f"Chunk {i+1} probabilities shape: {probs.shape}")
                probs_sum = probs if probs_sum is None else (probs_sum + probs)

            if probs_sum is None:
                return ("", 0.0, {}, 0)
                
            probs_mean = probs_sum / len(chunks)
            top_idx = int(np.argmax(probs_mean))
            top_label = self.id2label.get(top_idx, str(top_idx))
            top_conf = float(probs_mean[top_idx])
            probs_dict = {self.id2label[i]: float(p) for i, p in enumerate(probs_mean)}
            return top_label, top_conf, probs_dict, len(chunks)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return ("", 0.0, {}, 0)
