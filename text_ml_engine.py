#!/usr/bin/env python3
"""
Text-Based ML Engine - Lightweight Deep Text Processing
Uses open-source models optimized for laptop performance
"""

import json
import numpy as np
import re
import pickle
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter
import statistics
import math

# Lightweight text processing libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available. Install with: pip install scikit-learn")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[WARNING] NLTK not available. Install with: pip install nltk")

class TextMLEngine:
    """Lightweight text-based ML engine for deep text processing"""
    
    def __init__(self, data_file: str = "anonymous_data.json"):
        self.data_file = data_file
        self.text_data = []
        self.text_models = {}
        self.embeddings_cache = {}
        
        # Model configurations for laptop performance
        self.model_configs = {
            'sentence_transformer': 'all-MiniLM-L6-v2',  # Lightweight, 22MB
            'max_text_length': 512,
            'batch_size': 16,
            'cache_embeddings': True
        }
        
        # Initialize text processing
        self._initialize_text_processing()
        self._load_text_data()
        self._initialize_models()
    
    def _initialize_text_processing(self):
        """Initialize text preprocessing components"""
        print("🔤 Initializing text processing components...")
        
        # Initialize sentence transformer (lightweight model)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.model_configs['sentence_transformer'])
                print(f"[OK] Loaded {self.model_configs['sentence_transformer']} model")
            except Exception as e:
                print(f"[WARNING] Could not load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                
                self.stop_words = set(stopwords.words('english'))
                self.lemmatizer = WordNetLemmatizer()
                print("[OK] NLTK components initialized")
            except Exception as e:
                print(f"[WARNING] NLTK initialization failed: {e}")
                self.stop_words = set()
                self.lemmatizer = None
        else:
            self.stop_words = set()
            self.lemmatizer = None
        
        # Initialize TF-IDF vectorizer
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            print("[OK] TF-IDF vectorizer initialized")
        else:
            self.tfidf_vectorizer = None
    
    def _load_text_data(self):
        """Load and preprocess text data from sessions"""
        print("📚 Loading text data for training...")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sessions = data.get('sessions', [])
                
                for session in sessions:
                    text_data = self._extract_text_from_session(session)
                    if text_data:
                        self.text_data.append(text_data)
                
                print(f"[OK] Loaded {len(self.text_data)} text samples")
                
        except Exception as e:
            print(f"[WARNING] Could not load text data: {e}")
            self.text_data = []
    
    def _extract_text_from_session(self, session: Dict) -> Optional[Dict]:
        """Extract and combine text fields from a session"""
        text_fields = [
            session.get('current_mood', ''),
            session.get('emotional_state', ''),
            session.get('challenges', ''),
            session.get('goals', ''),
            session.get('life_changes', ''),
            session.get('coping_mechanisms', ''),
            session.get('previous_counseling', ''),
            session.get('medication', ''),
            session.get('support_system', '')
        ]
        
        # Combine all text fields
        combined_text = ' '.join([str(field).strip() for field in text_fields if field and str(field).strip()])
        
        if len(combined_text.strip()) < 10:  # Skip very short texts
            return None
        
        # Extract numeric features for labeling
        stress = session.get('stress_level', 5)
        sleep = session.get('sleep_quality', 5)
        social = session.get('social_support', 5)
        emotional = session.get('emotional_score', 50)
        
        return {
            'text': combined_text,
            'stress_level': stress,
            'sleep_quality': sleep,
            'social_support': social,
            'emotional_score': emotional,
            'user_id': session.get('user_id', 'anonymous'),
            'timestamp': session.get('timestamp', ''),
            'risk_level': self._calculate_risk_level(stress, sleep, social, emotional)
        }
    
    def _calculate_risk_level(self, stress: float, sleep: float, social: float, emotional: float) -> str:
        """Calculate risk level from numeric features"""
        if emotional < 30 or stress >= 8:
            return 'high'
        elif emotional < 60 or stress >= 6:
            return 'medium'
        else:
            return 'low'
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better model performance"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stop words if NLTK is available
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                tokens = word_tokenize(text)
                tokens = [token for token in tokens if token not in self.stop_words]
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                text = ' '.join(tokens)
            except Exception as e:
                print(f"[WARNING] NLTK preprocessing failed: {e}")
        
        return text
    
    def _initialize_models(self):
        """Initialize text-based ML models"""
        print("🤖 Initializing text-based ML models...")
        
        if len(self.text_data) < 10:
            print("[WARNING] Insufficient text data for model training")
            return
        
        # Prepare training data
        texts = [self._preprocess_text(item['text']) for item in self.text_data]
        risk_labels = [item['risk_level'] for item in self.text_data]
        stress_labels = [item['stress_level'] for item in self.text_data]
        sleep_labels = [item['sleep_quality'] for item in self.text_data]
        social_labels = [item['social_support'] for item in self.text_data]
        
        # 1. Risk Classification Model
        if SKLEARN_AVAILABLE:
            try:
                # TF-IDF features
                tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
                
                # Risk classification
                self.text_models['risk_classifier'] = MultinomialNB()
                self.text_models['risk_classifier'].fit(tfidf_features, risk_labels)
                
                # Stress level prediction
                self.text_models['stress_predictor'] = LogisticRegression(max_iter=1000)
                self.text_models['stress_predictor'].fit(tfidf_features, stress_labels)
                
                # Sleep quality prediction
                self.text_models['sleep_predictor'] = LogisticRegression(max_iter=1000)
                self.text_models['sleep_predictor'].fit(tfidf_features, sleep_labels)
                
                # Social support prediction
                self.text_models['social_predictor'] = LogisticRegression(max_iter=1000)
                self.text_models['social_predictor'].fit(tfidf_features, social_labels)
                
                print("[OK] Text-based classification models trained")
                
            except Exception as e:
                print(f"[WARNING] Text model training failed: {e}")
        
        # 2. Text Embeddings and Clustering
        if self.sentence_model:
            try:
                # Generate embeddings for all texts
                embeddings = self.sentence_model.encode(texts, batch_size=self.model_configs['batch_size'])
                
                # Text clustering
                n_clusters = min(8, max(3, len(texts) // 10))
                self.text_models['text_clustering'] = KMeans(n_clusters=n_clusters, random_state=42)
                self.text_models['text_clustering'].fit(embeddings)
                
                # Store embeddings for similarity search
                self.text_embeddings = embeddings
                self.text_clusters = self.text_models['text_clustering'].labels_
                
                print("[OK] Text embeddings and clustering models trained")
                
            except Exception as e:
                print(f"[WARNING] Embedding model training failed: {e}")
        
        # 3. Topic Modeling (if enough data)
        if len(texts) >= 50 and SKLEARN_AVAILABLE:
            try:
                # Simple topic modeling using TF-IDF
                self.text_models['topic_model'] = KMeans(n_clusters=min(10, len(texts) // 5), random_state=42)
                self.text_models['topic_model'].fit(tfidf_features)
                
                print("[OK] Topic modeling completed")
                
            except Exception as e:
                print(f"[WARNING] Topic modeling failed: {e}")
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text using all available models"""
        if not text or len(text.strip()) < 5:
            return self._get_empty_analysis()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        analysis = {
            'original_text': text,
            'processed_text': processed_text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Risk Classification
        if 'risk_classifier' in self.text_models and self.tfidf_vectorizer:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([processed_text])
                risk_prediction = self.text_models['risk_classifier'].predict(tfidf_features)[0]
                risk_probabilities = self.text_models['risk_classifier'].predict_proba(tfidf_features)[0]
                
                analysis['risk_prediction'] = risk_prediction
                analysis['risk_probabilities'] = {
                    'low': float(risk_probabilities[0]) if len(risk_probabilities) > 0 else 0.0,
                    'medium': float(risk_probabilities[1]) if len(risk_probabilities) > 1 else 0.0,
                    'high': float(risk_probabilities[2]) if len(risk_probabilities) > 2 else 0.0
                }
            except Exception as e:
                print(f"[WARNING] Risk classification failed: {e}")
        
        # 2. Numeric Predictions
        if self.tfidf_vectorizer:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([processed_text])
                
                if 'stress_predictor' in self.text_models:
                    stress_pred = self.text_models['stress_predictor'].predict(tfidf_features)[0]
                    analysis['predicted_stress'] = int(stress_pred)
                
                if 'sleep_predictor' in self.text_models:
                    sleep_pred = self.text_models['sleep_predictor'].predict(tfidf_features)[0]
                    analysis['predicted_sleep'] = int(sleep_pred)
                
                if 'social_predictor' in self.text_models:
                    social_pred = self.text_models['social_predictor'].predict(tfidf_features)[0]
                    analysis['predicted_social'] = int(social_pred)
                    
            except Exception as e:
                print(f"[WARNING] Numeric prediction failed: {e}")
        
        # 3. Text Embeddings and Similarity
        if self.sentence_model:
            try:
                # Generate embedding for input text
                text_embedding = self.sentence_model.encode([processed_text])[0]
                
                # Find similar texts
                if hasattr(self, 'text_embeddings'):
                    similarities = cosine_similarity([text_embedding], self.text_embeddings)[0]
                    most_similar_idx = np.argmax(similarities)
                    analysis['similarity_score'] = float(similarities[most_similar_idx])
                    analysis['most_similar_text'] = self.text_data[most_similar_idx]['text'][:200] + "..."
                
                # Text clustering
                if 'text_clustering' in self.text_models:
                    cluster = self.text_models['text_clustering'].predict([text_embedding])[0]
                    analysis['text_cluster'] = int(cluster)
                    
            except Exception as e:
                print(f"[WARNING] Embedding analysis failed: {e}")
        
        # 4. Sentiment Analysis (simple rule-based)
        analysis['sentiment'] = self._analyze_sentiment(processed_text)
        
        # 5. Emotion Keywords Detection
        analysis['emotions'] = self._detect_emotions(processed_text)
        
        # 6. Topic Classification
        if 'topic_model' in self.text_models and self.tfidf_vectorizer:
            try:
                tfidf_features = self.tfidf_vectorizer.transform([processed_text])
                topic = self.text_models['topic_model'].predict(tfidf_features)[0]
                analysis['topic'] = int(topic)
            except Exception as e:
                print(f"[WARNING] Topic classification failed: {e}")
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Simple rule-based sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'positive', 'better', 'improved', 'well', 'fine', 'okay']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'negative', 'worse', 'depressed', 'anxious', 'stressed', 'worried']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        positive_ratio = positive_count / total_sentiment_words
        
        if positive_ratio > 0.6:
            sentiment = 'positive'
        elif positive_ratio < 0.4:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(positive_ratio - 0.5) * 2,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _detect_emotions(self, text: str) -> Dict:
        """Detect emotions based on keyword matching"""
        emotion_keywords = {
            'anxiety': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared'],
            'depression': ['sad', 'depressed', 'hopeless', 'empty', 'worthless', 'suicidal'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'irritated', 'frustrated'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'tension', 'burnout'],
            'loneliness': ['lonely', 'isolated', 'alone', 'disconnected', 'empty'],
            'happiness': ['happy', 'joy', 'excited', 'pleased', 'content', 'grateful']
        }
        
        emotions = {}
        words = text.lower().split()
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in words)
            emotions[emotion] = count
        
        return emotions
    
    def _get_empty_analysis(self) -> Dict:
        """Return empty analysis for invalid input"""
        return {
            'original_text': '',
            'processed_text': '',
            'text_length': 0,
            'word_count': 0,
            'timestamp': datetime.now().isoformat(),
            'error': 'Insufficient text for analysis'
        }
    
    def get_text_insights(self, user_data: Dict) -> Dict:
        """Get comprehensive text insights for user data"""
        # Combine all text fields
        text_fields = [
            user_data.get('current_mood', ''),
            user_data.get('emotional_state', ''),
            user_data.get('challenges', ''),
            user_data.get('goals', ''),
            user_data.get('life_changes', ''),
            user_data.get('coping_mechanisms', ''),
            user_data.get('previous_counseling', ''),
            user_data.get('medication', ''),
            user_data.get('support_system', '')
        ]
        
        combined_text = ' '.join([str(field).strip() for field in text_fields if field and str(field).strip()])
        
        if len(combined_text.strip()) < 10:
            return {'error': 'Insufficient text data for analysis'}
        
        # Analyze combined text
        analysis = self.analyze_text(combined_text)
        
        # Add individual field analysis
        field_analyses = {}
        for field_name, field_text in zip(['current_mood', 'emotional_state', 'challenges', 'goals', 'life_changes', 
                                         'coping_mechanisms', 'previous_counseling', 'medication', 'support_system'], text_fields):
            if field_text and len(str(field_text).strip()) > 5:
                field_analyses[field_name] = self.analyze_text(str(field_text))
        
        return {
            'combined_analysis': analysis,
            'field_analyses': field_analyses,
            'text_summary': {
                'total_text_length': len(combined_text),
                'fields_with_text': len([f for f in text_fields if f and str(f).strip()]),
                'analysis_confidence': analysis.get('similarity_score', 0.0)
            }
        }
    
    def add_text_session(self, session_data: Dict):
        """Add new text session for incremental learning"""
        text_data = self._extract_text_from_session(session_data)
        if text_data:
            self.text_data.append(text_data)
            
            # Update models if we have enough new data
            if len(self.text_data) % 10 == 0:  # Update every 10 new sessions
                self._update_models_incrementally()
    
    def _update_models_incrementally(self):
        """Update models with new text data"""
        print("[UPDATE] Updating text models incrementally...")
        
        if len(self.text_data) < 20:
            return
        
        # Retrain models with all data
        self._initialize_models()
        print("[OK] Text models updated with new data")
    
    def get_model_status(self) -> Dict:
        """Get status of all text models"""
        return {
            'total_text_samples': len(self.text_data),
            'models_available': list(self.text_models.keys()),
            'sentence_transformer': self.sentence_model is not None,
            'sklearn_available': SKLEARN_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE,
            'last_updated': datetime.now().isoformat(),
            'model_config': self.model_configs
        }


# Example usage and testing
if __name__ == "__main__":
    print("[START] Initializing Text-Based ML Engine...")
    
    # Initialize the engine
    text_engine = TextMLEngine()
    
    # Test with sample text
    test_text = "I feel very anxious and overwhelmed with work. I'm having trouble sleeping and feel isolated from my friends. I want to get better at managing stress and improving my relationships."
    
    print("\n📝 Testing text analysis...")
    analysis = text_engine.analyze_text(test_text)
    
    print("\n[TARGET] Text Analysis Results:")
    print("=" * 50)
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n[DATA] Model Status:")
    status = text_engine.get_model_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n[OK] Text-based ML engine ready for integration!")
