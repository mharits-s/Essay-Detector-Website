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
        logger.info("Initializing TextClassifier...")
        self.model_dir = model_dir

        tokenizer_sen_path = 'models_15jun/ta_sentence_1/tokenizer'
        semantic_sen_model = 'models_15jun/ta_sentence_1/semantic_model.h5'

        tokenizer_par_path = 'models_15jun/ta_paragraph_1/tokenizer'
        semantic_par_model = 'models_15jun/ta_paragraph_1/semantic_model.h5'
        
        custom_objects = {
            'TFBertModel': TFBertModel,
            'Adam': tf.keras.optimizers.Adam(learning_rate=1e-4)
        }

        def load_model_safely(model_path, model_name):
            """
            Safely load model with fallback for optimizer compatibility issues
            """
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info(f"{model_name} loaded successfully")
                return model
                
            except TypeError as e:
                if "weight_decay" in str(e):
                    logger.warning(f"Optimizer compatibility issue with {model_name}, trying without compile...")
                    try:
                        model = tf.keras.models.load_model(model_path, compile=False)
 
                        model.compile(
                            optimizer='adam', 
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        logger.info(f"{model_name} loaded and recompiled successfully")
                        return model
                        
                    except Exception as e2:
                        logger.error(f"Failed to load {model_name}: {e2}")
                        raise
                else:
                    logger.error(f"Failed to load {model_name}: {e}")
                    raise
            
            except Exception as e:
                logger.error(f"Unexpected error loading {model_name}: {e}")
                raise

        # Loading models dengan custom object scope
        with tf.keras.utils.custom_object_scope(custom_objects):
            try:
                # Load semantic models
                self.semantic_sen_model = load_model_safely(
                    semantic_sen_model, 
                    "Semantic sentence model"
                )
                
                self.semantic_par_model = load_model_safely(
                    semantic_par_model,
                    "Semantic paragraph model"
                )
                
                self.classifier_sen_model = load_model_safely(
                    f'{model_dir}/ta_sentence_1/classification_model.h5',
                    "Classification sentence model"
                )
                
                self.classifier_par_model = load_model_safely(
                    f'{model_dir}/ta_paragraph_1/classification_model.h5',
                    "Classification paragraph model"
                )
                
                logger.info("All models loaded successfully!")
                
            except Exception as e:
                logger.error(f"Critical error during model loading: {e}")
                raise RuntimeError(f"Failed to initialize models: {e}")

        try:
            # Check if required files exist
            required_files = [
                tokenizer_sen_path,
                semantic_sen_model,
                tokenizer_par_path,
                semantic_par_model,
                f'{model_dir}/ta_sentence_1/classification_model.h5',
                f'{model_dir}/ta_paragraph_1/classification_model.h5',
                f'{model_dir}/ta_sentence_1/scaler_linguistic.pkl',
                f'{model_dir}/ta_paragraph_1/scaler_linguistic.pkl',
                f'{model_dir}/ta_sentence_1/tokenized_data.pkl',
                f'{model_dir}/ta_paragraph_1/tokenized_data.pkl',
                f'{model_dir}/ta_sentence_1/reference_embeddings.pkl',
                f'{model_dir}/ta_paragraph_1/reference_embeddings.pkl',
            ]

            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.error(f"Required file not found: {file_path}")
                    raise FileNotFoundError(f"Required file not found: {file_path}")
                else:
                    logger.info(f"Found required file: {file_path}")
            
            # Load tokenizer
            logger.info("Loading tokenizers...")
            self.tokenizer_sen = BertTokenizer.from_pretrained(tokenizer_sen_path)
            self.tokenizer_par = BertTokenizer.from_pretrained(tokenizer_par_path)
            logger.info("Tokenizers loaded successfully")

            # Load scaler for linguistic features
            logger.info("Loading scalers and tokenized data...")
            with open(f'{model_dir}/ta_sentence_1/scaler_linguistic.pkl', "rb") as f:
                self.scaler_sen = pickle.load(f)

            with open(f'{model_dir}/ta_paragraph_1/scaler_linguistic.pkl', "rb") as f:
                self.scaler_par = pickle.load(f)

            with open(f'{model_dir}/ta_sentence_1/reference_embeddings.pkl', 'rb') as f:
                self.sentence_embeddings = pickle.load(f)

            with open(f'{model_dir}/ta_paragraph_1/reference_embeddings.pkl', 'rb') as f:
                self.paragraph_embeddings = pickle.load(f)

            logger.info("TextClassifier initialization completed successfully")
                
        except Exception as e:
            logger.error(f"Error during TextClassifier initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

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
            chatgpt_similarity = self.cos_sim(embedding, self.sentence_embeddings['embeddings_gpt_sen']['embeddings'])

            # Combine similarity scores
            similarity_scores = np.array([[
                student_similarity,
                chatgpt_similarity
            ]])
            
            # Prepare inputs for classifier
            inputs = {
                "embeddings": embedding,
                "similarity_score": similarity_scores,
                "linguistic_features": normalized_features
            }
            
            # Classify
            prediction = self.classifier_sen_model.predict(inputs, verbose=0)[0][0]
            class_label = "AI" if prediction > 0.6140 else "Human"
            
            logger.info(f"Sentence classified as {class_label} with probability {prediction:.4f}")
            
            return prediction, class_label, {
                "student_similarity": student_similarity,
                "chatgpt_similarity": chatgpt_similarity,
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
            chatgpt_similarity = self.cos_sim(embedding, self.paragraph_embeddings['embeddings_gpt_par']['embeddings'])

            # Combine similarity scores
            similarity_scores = np.array([[
                student_similarity,
                chatgpt_similarity
            ]])
            
            # Prepare inputs for classifier
            inputs = {
                "embeddings": embedding,
                "similarity_score": similarity_scores,
                "linguistic_features": normalized_features
            }
            
            # Classify
            prediction = self.classifier_par_model.predict(inputs, verbose=0)[0][0]
            class_label = "AI" if prediction > 0.5522 else "Human"
            
            logger.info(f"Paragraph classified as {class_label} with probability {prediction:.4f}")
            
            return prediction, class_label, {
                "student_similarity": student_similarity,
                "chatgpt_similarity": chatgpt_similarity,
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
    return render_template('input.html')

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
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(port=5001)