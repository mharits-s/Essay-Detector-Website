from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import pickle
import re
import os
import numpy as np
from transformers import BertTokenizer, TFBertModel
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import logging
import traceback
import json
from google.cloud import storage

SA_INFO = json.loads(os.getenv("GCP_SA_JSON"))
GCS_BUCKET = os.getenv("GCS_BUCKET")
MODELS_PREFIX = os.getenv("MODELS_PREFIX", "models_15jun/")
TMP_DIR = "/tmp"

gcs_client = storage.Client.from_service_account_info(SA_INFO)
bucket = gcs_client.bucket(GCS_BUCKET)

def _gcs_to_tmp(gcs_key: str, local_rel_path: str) -> str:
    """
    Unduh satu file dari GCS ke /tmp/<local_rel_path> bila belum ada.
    Return: absolute local path.
    """
    local_path = os.path.join(TMP_DIR, local_rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        blob = bucket.blob(gcs_key)
        blob.download_to_filename(local_path)
    return local_path

def _gcs_dir_to_tmp(gcs_dir_prefix: str, local_rel_dir: str) -> str:
    """
    Unduh satu folder (mis. tokenizer/) dari GCS ke /tmp secara rekursif.
    Return: absolute local dir path.
    """
    local_dir = os.path.join(TMP_DIR, local_rel_dir)
    os.makedirs(local_dir, exist_ok=True)
    # list blobs di prefix
    for blob in gcs_client.list_blobs(GCS_BUCKET, prefix=gcs_dir_prefix):
        # lewati "folder marker"
        if blob.name.endswith("/"):
            continue
        rel = os.path.relpath(blob.name, gcs_dir_prefix)
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            blob.download_to_filename(dest)
    return local_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TextClassifier:
    def __init__(self, model_dir='models_15jun'):
        """
        Initialize the text classifier with pre-trained models_15jun.
        
        Args:
            model_dir (str): Directory containing saved models_15jun and configurations.
        """
        logger.info("Initializing TextClassifier with GCS...")
        sen_dir = f"{MODELS_PREFIX}ta_sentence_2/"
        par_dir = f"{MODELS_PREFIX}ta_paragraph_2/"

        # File2 utama (H5/PKL)
        paths = {
            "semantic_sen_h5":   sen_dir + "semantic_model.h5",
            "cls_sen_h5":        sen_dir + "classification_model.h5",
            "scaler_sen_pkl":    sen_dir + "scaler_linguistic.pkl",
            "ref_emb_sen_pkl":   sen_dir + "reference_embeddings.pkl",
            "tokdata_sen_pkl":   sen_dir + "tokenized_data.pkl", 

            "semantic_par_h5":   par_dir + "semantic_model.h5",
            "cls_par_h5":        par_dir + "classification_model.h5",
            "scaler_par_pkl":    par_dir + "scaler_linguistic.pkl",
            "ref_emb_par_pkl":   par_dir + "reference_embeddings.pkl",
            "tokdata_par_pkl":   par_dir + "tokenized_data.pkl", 
        }

        # Tokenizer (ini berupa folder)
        tok_sen_gcs_dir = sen_dir + "tokenizer/"
        tok_par_gcs_dir = par_dir + "tokenizer/"

        # ==== unduh ke /tmp ====
        local_semantic_sen = _gcs_to_tmp(paths["semantic_sen_h5"], "models_15jun/ta_sentence_2/semantic_model.h5")
        local_cls_sen      = _gcs_to_tmp(paths["cls_sen_h5"],      "models_15jun/ta_sentence_2/classification_model.h5")
        local_scaler_sen   = _gcs_to_tmp(paths["scaler_sen_pkl"],  "models_15jun/ta_sentence_2/scaler_linguistic.pkl")
        local_ref_sen      = _gcs_to_tmp(paths["ref_emb_sen_pkl"], "models_15jun/ta_sentence_2/reference_embeddings.pkl")
        local_tokdata_sen = _gcs_to_tmp(paths["tokdata_sen_pkl"], "models_15jun/ta_sentence_2/tokenized_data.pkl")

        local_semantic_par = _gcs_to_tmp(paths["semantic_par_h5"], "models_15jun/ta_paragraph_2/semantic_model.h5")
        local_cls_par      = _gcs_to_tmp(paths["cls_par_h5"],      "models_15jun/ta_paragraph_2/classification_model.h5")
        local_scaler_par   = _gcs_to_tmp(paths["scaler_par_pkl"],  "models_15jun/ta_paragraph_2/scaler_linguistic.pkl")
        local_ref_par      = _gcs_to_tmp(paths["ref_emb_par_pkl"], "models_15jun/ta_paragraph_2/reference_embeddings.pkl")
        local_tokdata_par = _gcs_to_tmp(paths["tokdata_par_pkl"], "models_15jun/ta_paragraph_2/tokenized_data.pkl")

        # tokenizer folders -> /tmp/...
        local_tok_sen_dir = _gcs_dir_to_tmp(tok_sen_gcs_dir, "models_15jun/ta_sentence_2/tokenizer")
        local_tok_par_dir = _gcs_dir_to_tmp(tok_par_gcs_dir, "models_15jun/ta_paragraph_2/tokenizer")

        
        custom_objects = {
            'TFBertModel': TFBertModel,
            'Adam': tf.keras.optimizers.Adam(learning_rate=1e-4)
        }

        def load_model_safely(model_path, model_name):
            try:
                return tf.keras.models.load_model(model_path)
            except TypeError as e:
                if "weight_decay" in str(e):
                    m = tf.keras.models.load_model(model_path, compile=False)
                    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return m
                raise
        with tf.keras.utils.custom_object_scope(custom_objects):
            self.semantic_sen_model = load_model_safely(local_semantic_sen, "Semantic sentence model")
            self.semantic_par_model = load_model_safely(local_semantic_par, "Semantic paragraph model")
            self.classifier_sen_model = load_model_safely(local_cls_sen, "Classification sentence model")
            self.classifier_par_model = load_model_safely(local_cls_par, "Classification paragraph model")

        # ==== load tokenizer & pkl dari /tmp ====
        self.tokenizer_sen = BertTokenizer.from_pretrained(local_tok_sen_dir)
        self.tokenizer_par = BertTokenizer.from_pretrained(local_tok_par_dir)

        import pickle
        with open(local_scaler_sen, "rb") as f: self.scaler_sen = pickle.load(f)
        with open(local_scaler_par, "rb") as f: self.scaler_par = pickle.load(f)
        with open(local_ref_sen, "rb") as f: self.sentence_embeddings = pickle.load(f)
        with open(local_ref_par, "rb") as f: self.paragraph_embeddings = pickle.load(f)
        with open(local_tokdata_sen, "rb") as f: self.tokenized_data_sen = pickle.load(f)
        with open(local_tokdata_par, "rb") as f: self.tokenized_data_par = pickle.load(f)

        logger.info("All models, tokenizers, and PKLs loaded from /tmp successfully.")

    def preprocess_sentence(self, text):
        """
        Text preprocessing:
        - Convert text to lowercase
        - Split text into sentences using regex
        - Keep final punctuation marks (., ?, !)

        Args:
            text (str): Input text.

        Returns:
            list: List of processed sentences.
        """
        logger.info(f"Preprocessing sentence: {text[:50]}...")
        
        if not isinstance(text, str) or text.strip() == "":
            logger.warning("Empty or invalid text provided")
            return []

        text = text.lower().strip()
        sentences = re.findall(r'[^.!?]+[.!?]?', text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        logger.info(f"Preprocessed into {len(sentences)} sentences")
        return sentences

    def preprocess_paragraph(self, text):
        """
        Text preprocessing:
        - Convert text to lowercase
        - Clean excess whitespace
        - Split text into paragraphs
        - Return list of cleaned paragraphs
        
        Args:
            text (str): Input text.
            
        Returns:
            list: List of processed paragraphs or empty list if invalid.
        """
        logger.info(f"Preprocessing paragraph: {text[:50]}...")
        
        if not isinstance(text, str) or text.strip() == "":
            logger.warning("Empty or invalid text provided")
            return []

        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]

        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.lower().strip()
            para = re.sub(r'\s+', ' ', para)
            if para:
                cleaned_paragraphs.append(para)
        
        logger.info(f"Preprocessed into {len(cleaned_paragraphs)} paragraphs")
        return cleaned_paragraphs

    def tokenize_sentence(self, texts, max_length=128):
        """
        Text tokenization using IndoBERT tokenizer.
        
        Args:
            texts (list): List of texts to be tokenized.
            max_length (int): Maximum token length.
            
        Returns:
            dict: Tokenized result, including input_ids and attention_mask.
        """
        logger.info(f"Tokenizing {len(texts)} sentences...")
        return self.tokenizer_sen(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )

    def tokenize_paragraph(self, texts, max_length=256):
        """
        Text tokenization using IndoBERT tokenizer.
        
        Args:
            texts (list): List of texts to be tokenized.
            max_length (int): Maximum token length.
            
        Returns:
            dict: Tokenized result, including input_ids and attention_mask.
        """
        logger.info(f"Tokenizing {len(texts)} paragraphs...")
        return self.tokenizer_par(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="tf"
        )

    def linguistic_features(self, text):
        """
        Features extraction from text:
        1.	Lexical Diversity
        2.	Total words in the essay
        3.	Total unique words*
        4.	Modals
        5.	Stopwords ratio*
        6.	Average sentence length*
        7.	Sentence length variation*
        8.	Punctuation Ratio*

        
        Args:
            text (str): Input text.
            
        Returns:
            dict: Linguistic features.
        """
        logger.info("Extracting linguistic features...")
        
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        unique_count = len(set(words))
        
        ld = (unique_count / word_count * 100) if word_count > 0 else 0
        
        # Load modals from corpus file
        modals = set()
        if os.path.exists('corpus/Indonesian_Manually_Tagged_Corpus_ID.tsv'):
            with open('corpus/Indonesian_Manually_Tagged_Corpus_ID.tsv', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and parts[1] == 'MD':
                            modals.add(parts[0].lower())
        
        # Count modals in text
        modal_count = sum(1 for word in words if word.lower() in modals)
        
        # Load stopwords from file
        stopwords = set()
        if os.path.exists('corpus/stopwords.txt'):
            with open('corpus/stopwords.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip())
        
        # Calculate stopword ratio
        stopword_count = sum(1 for word in words if word.lower() in stopwords)
        stopword_ratio = (stopword_count / word_count * 100) if word_count > 0 else 0
        
        # Calculate sentence length statistics
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        avg_sent_len = np.mean(sentence_lengths) if sentence_lengths else 0
        sent_len_var = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Calculate punctuation ratio
        punct_count = len(re.findall(r'[.!?]', text))
        punct_ratio = (punct_count / word_count) * 100 if word_count > 0 else 0

        features = {
            'lexical_diversity': ld,
            'total_words': word_count,
            'total_unique_words': unique_count,
            'modals': modal_count,
            'stopwords_ratio': stopword_ratio,
            'avg_sentence_length': avg_sent_len,
            'sentence_length_variation': sent_len_var,
            'punctuation_ratio': punct_ratio
        }
        
        logger.info(f"Extracted features: {features}")
        return features

    def gen_emb(self, tokens, model, batch_size=32):
        """
        Optimized embedding generation with larger batch size and better memory management.
        """
        logger.info(f"Generating embeddings for {len(tokens['input_ids'])} tokens...")
        
        embeddings = []
        num_samples = len(tokens['input_ids'])
        
        # Use larger batch size for better GPU utilization
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            batch_input_ids = tokens['input_ids'][i:end_idx]
            batch_attention_mask = tokens['attention_mask'][i:end_idx]
            
            # Generate embeddings for batch
            batch_embeddings = model([batch_input_ids, batch_attention_mask])
            embeddings.append(batch_embeddings.numpy())
            
            # Clear memory periodically
            if i % (batch_size * 10) == 0:
                tf.keras.backend.clear_session()
        
        result = np.concatenate(embeddings, axis=0)
        logger.info(f"Generated embeddings shape: {result.shape}")
        return result
    
    def cos_sim(self, embedding1, embedding2):
        """
        Optimized cosine similarity calculation using vectorized operations.
        """
        # Normalize embeddings
        embedding1_norm = tf.nn.l2_normalize(embedding1, axis=-1)
        embedding2_norm = tf.nn.l2_normalize(embedding2, axis=-1)
        
        # Calculate similarity matrix
        similarities = tf.matmul(embedding1_norm, tf.transpose(embedding2_norm))
        
        # Return average similarity
        return tf.reduce_mean(similarities).numpy()

    def classify_sentence(self, sentence):
        """
        Optimized sentence classification using pre-computed reference embeddings.
        """
        logger.info(f"Classifying sentence: {sentence[:50]}...")
        
        try:
            # Preprocess sentence
            processed_sentences = self.preprocess_sentence(sentence)
            if not processed_sentences:
                return None, None, None
            
            processed_sentence = " ".join(processed_sentences)
            
            # Tokenize sentence
            tokens = self.tokenize_sentence([processed_sentence])
            
            # Generate embedding for input only
            embedding = self.gen_emb(tokens, self.semantic_sen_model)
            
            # Extract and normalize linguistic features
            style_features = self.linguistic_features(processed_sentence)
            style_features_df = np.array([[
                style_features['lexical_diversity'],
                style_features['total_words'],
                style_features['total_unique_words'],
                style_features['modals'],
                style_features['stopwords_ratio'],
            ]])
            normalized_features = self.scaler_sen.transform(style_features_df)

            # Use pre-computed reference embeddings
            student_similarity = self.cos_sim(embedding, self.sentence_embeddings['embeddings_std_sen']['embeddings'])
            chatgpt_similarity = self.cos_sim(embedding, self.sentence_embeddings['embeddings_gpt1_sen']['embeddings'])
            chatgpt_knowledge_similarity = self.cos_sim(embedding, self.sentence_embeddings['embeddings_gpt2_sen']['embeddings'])

            # Combine similarity scores
            similarity_scores = np.array([[
                student_similarity,
                chatgpt_similarity,
                chatgpt_knowledge_similarity
            ]])
            
            # Prepare inputs for classifier
            inputs = {
                "embeddings": embedding,
                "similarity_score": similarity_scores,
                "linguistic_features": normalized_features
            }
            
            # Classify
            prediction = self.classifier_sen_model.predict(inputs, verbose=0)[0][0]
            class_label = "AI" if prediction > 0.2799 else "Human"
            
            logger.info(f"Sentence classified as {class_label} with probability {prediction:.4f}")
            
            return prediction, class_label, {
                "student_similarity": student_similarity,
                "chatgpt_similarity": chatgpt_similarity,
                "chatgpt_knowledge_similarity": chatgpt_knowledge_similarity,
                "linguistic_features": style_features
            }
            
        except Exception as e:
            logger.error(f"Error classifying sentence: {str(e)}")
            return None, None, None

    def classify_paragraph(self, paragraph):
        """
        Optimized paragraph classification using pre-computed reference embeddings.
        """
        logger.info(f"Classifying paragraph: {paragraph[:50]}...")
        
        try:
            # Preprocess paragraph
            processed_paragraphs = self.preprocess_paragraph(paragraph)
            if not processed_paragraphs:
                return None, None, None
            
            processed_paragraph = " ".join(processed_paragraphs)
            
            # Tokenize paragraph
            tokens = self.tokenize_paragraph([processed_paragraph])

            # Generate embedding for input only
            embedding = self.gen_emb(tokens, self.semantic_par_model)
            
            # Extract and normalize linguistic features
            style_features = self.linguistic_features(processed_paragraph)
            style_features_df = np.array([[
                style_features['lexical_diversity'],
                style_features['total_words'],
                style_features['total_unique_words'],
                style_features['modals'],
                style_features['stopwords_ratio'],
                style_features['avg_sentence_length'],
                style_features['sentence_length_variation'],
                style_features['punctuation_ratio']
            ]])
            normalized_features = self.scaler_par.transform(style_features_df)

            student_similarity = self.cos_sim(embedding, self.paragraph_embeddings['embeddings_std_par']['embeddings'])
            chatgpt_similarity = self.cos_sim(embedding, self.paragraph_embeddings['embeddings_gpt1_par']['embeddings'])
            chatgpt_knowledge_similarity = self.cos_sim(embedding, self.paragraph_embeddings['embeddings_gpt2_par']['embeddings'])

            # Combine similarity scores
            similarity_scores = np.array([[
                student_similarity,
                chatgpt_similarity,
                chatgpt_knowledge_similarity
            ]])
            
            # Prepare inputs for classifier
            inputs = {
                "embeddings": embedding,
                "similarity_score": similarity_scores,
                "linguistic_features": normalized_features
            }
            
            # Classify
            prediction = self.classifier_par_model.predict(inputs, verbose=0)[0][0]
            class_label = "AI" if prediction > 0.9170 else "Human"
            
            logger.info(f"Paragraph classified as {class_label} with probability {prediction:.4f}")
            
            return prediction, class_label, {
                "student_similarity": student_similarity,
                "chatgpt_similarity": chatgpt_similarity,
                "chatgpt_knowledge_similarity": chatgpt_knowledge_similarity,
                "linguistic_features": style_features
            }
            
        except Exception as e:
            logger.error(f"Error classifying paragraph: {str(e)}")
            return None, None, None
    
    def classify_text(self, text):
        """
        Classify text by analyzing both paragraphs and sentences.
        
        Args:
            text (str): Input text.
            
        Returns:
            dict: Classification results including both paragraph and sentence level analysis.
        """
        logger.info(f"Starting text classification for: {text[:100]}...")
        
        try:
            # Split text into paragraphs
            paragraphs = self.preprocess_paragraph(text)
            logger.info(f"Split into {len(paragraphs)} paragraphs")
            
            # Classify each paragraph
            paragraph_results = []
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    logger.info(f"Processing paragraph {i+1}/{len(paragraphs)}")
                    prob, label, features = self.classify_paragraph(paragraph)
                    if prob is not None:
                        paragraph_results.append({
                            "paragraph": paragraph,
                            "probability": prob,
                            "class": label,
                            "features": features
                        })
            
            # Calculate paragraph statistics
            total_paragraphs = len(paragraph_results)
            ai_paragraphs = sum(1 for r in paragraph_results if r["class"] == "AI")
            human_paragraphs = total_paragraphs - ai_paragraphs
            
            paragraph_ai_percentage = (ai_paragraphs / total_paragraphs) * 100 if total_paragraphs > 0 else 0
            paragraph_human_percentage = (human_paragraphs / total_paragraphs) * 100 if total_paragraphs > 0 else 0
            
            logger.info(f"Paragraph results: {ai_paragraphs} AI, {human_paragraphs} Human")
            
            # Split text into sentences
            sentences = self.preprocess_sentence(text)
            logger.info(f"Split into {len(sentences)} sentences")
            
            # Classify each sentence
            sentence_results = []
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    logger.info(f"Processing sentence {i+1}/{len(sentences)}")
                    prob, label, features = self.classify_sentence(sentence)
                    if prob is not None:
                        sentence_results.append({
                            "sentence": sentence,
                            "probability": prob,
                            "class": label,
                            "features": features
                        })
            
            # Calculate sentence statistics
            total_sentences = len(sentence_results)
            ai_sentences = sum(1 for r in sentence_results if r["class"] == "AI")
            human_sentences = total_sentences - ai_sentences
            
            sentence_ai_percentage = (ai_sentences / total_sentences) * 100 if total_sentences > 0 else 0
            sentence_human_percentage = (human_sentences / total_sentences) * 100 if total_sentences > 0 else 0
            
            logger.info(f"Sentence results: {ai_sentences} AI, {human_sentences} Human")
            
            # Sort sentences by probability for top lists
            ai_sorted = sorted([r for r in sentence_results if r["class"] == "AI"], 
                              key=lambda x: x["probability"], reverse=True)
            human_sorted = sorted([r for r in sentence_results if r["class"] == "Human"], 
                                 key=lambda x: 1 - x["probability"], reverse=True)
            
            # Get top 5 sentences for each class
            top_ai = ai_sorted[:5]
            top_human = human_sorted[:5]
            
            result = {
                "sentences": sentence_results,
                "paragraphs": paragraph_results,
                "statistics": {
                    # Sentence level statistics
                    "total_sentences": total_sentences,
                    "ai_sentences": ai_sentences,
                    "human_sentences": human_sentences,
                    "ai_percentage": sentence_ai_percentage,
                    "human_percentage": sentence_human_percentage,
                    # Paragraph level statistics
                    "total_paragraphs": total_paragraphs,
                    "ai_paragraphs": ai_paragraphs,
                    "human_paragraphs": human_paragraphs,
                    "paragraph_ai_percentage": paragraph_ai_percentage,
                    "paragraph_human_percentage": paragraph_human_percentage
                },
                "top_ai": top_ai,
                "top_human": top_human
            }
            
            logger.info("Text classification completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in classify_text: {str(e)}")
            logger.error(traceback.format_exc())
            raise

from functools import lru_cache
import hashlib

class CachedTextClassifier(TextClassifier):
    def __init__(self, model_dir='models_15jun', cache_size=128):
        super().__init__(model_dir)
        self.cache_size = cache_size
        
    @lru_cache(maxsize=128)
    def _cached_embedding(self, text_hash, text_type):
        """
        Cache embeddings for repeated texts to avoid recomputation.
        """
        # This would need to be implemented with actual caching logic
        pass
    
    def _get_text_hash(self, text):
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode()).hexdigest()

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

@app.route('/public/assets/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('public/assets/images', filename)

@app.route('/')
def home():
    logger.info("Home page accessed")
    return render_template('input_final.html')

# Initialize classifier with error handling
classifier = None
try:
    logger.info("Starting classifier initialization...")
    classifier = TextClassifier()
    logger.info("Classifier initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize classifier: {str(e)}")
    logger.error(traceback.format_exc())

@app.route('/output', methods=['POST'])
def classify():
    text_input = request.form['text_input']
    try:
        results = classifier.classify_text(text_input)
        return render_template('output_final.html', results=results, original_text=text_input)
    except Exception as e:
        return render_template('error.html', error_message=str(e)), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))