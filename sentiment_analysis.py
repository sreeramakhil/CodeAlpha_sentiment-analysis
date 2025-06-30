import re
from collections import Counter
from datetime import datetime

# Optional imports with fallbacks
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not found. Some features will be limited.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not found. Using basic Python operations.")

# Plotting libraries are excluded as the request is only for sentiment analysis code
# try:
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     HAS_PLOTTING = True
# except ImportError:
#     HAS_PLOTTING = False

# NLP Libraries (optional)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
    
    # Download required NLTK data (quiet=True to avoid verbose output during import)
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download NLTK data ({e}). Some NLP features may not work.")
        HAS_NLTK = False # Disable NLTK if data download fails
        
except ImportError:
    HAS_NLTK = False
    print("Warning: NLTK not found. Basic sentiment analysis will be used.")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("Warning: TextBlob not found. Basic sentiment analysis will be used.")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: Transformers not found. Advanced models disabled.")

# Web scraping libraries are excluded as the request is only for sentiment analysis code
# try:
#     import requests
#     from bs4 import BeautifulSoup
#     HAS_SCRAPING = True
# except ImportError:
#     HAS_SCRAPING = False

# try:
#     import tweepy # For Twitter API (requires API keys)
#     HAS_TWEEPY = True
# except ImportError:
#     HAS_TWEEPY = False


class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer with multiple models"""
        # Basic sentiment lexicon (built-in)
        self.positive_words = set([
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
            'perfect', 'love', 'like', 'best', 'brilliant', 'outstanding', 'superb',
            'happy', 'pleased', 'satisfied', 'delighted', 'thrilled', 'impressed'
        ])
        
        self.negative_words = set([
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'worst', 'poor', 'disappointing', 'frustrated', 'angry', 'upset', 'sad',
            'annoying', 'useless', 'broken', 'failed', 'problem', 'issue', 'complaint'
        ])

        if HAS_NLTK:
            self.sia = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.sia = None
            self.lemmatizer = None # Lemmatizer won't be available
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Load advanced transformer model
        if HAS_TRANSFORMERS:
            try:
                # Prioritize the specified model for Twitter sentiment
                self.transformer_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )                      
            except Exception as e:
                print(f"Warning: Could not load 'cardiffnlp/twitter-roberta-base-sentiment-latest': {e}")
                print("Attempting to load default 'sentiment-analysis' pipeline.")
                try:
                    self.transformer_analyzer = pipeline("sentiment-analysis")
                except Exception as e:
                    print(f"Warning: Could not load default 'sentiment-analysis' pipeline: {e}")
                    self.transformer_analyzer = None
        else:
            self.transformer_analyzer = None
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment_basic(self, text):
        """Basic sentiment analysis using TextBlob or simple lexicon fallback"""
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except Exception as e:
                print(f"TextBlob analysis failed: {e}. Falling back to basic lexicon.")
        
        # Fallback to simple lexicon if TextBlob fails or is not available
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            polarity = 0.0
            sentiment = 'neutral'
        else:
            polarity = (positive_count - negative_count) / total_sentiment_words # Ratio of sentiment words
            if polarity > 0.15: # Slightly more aggressive thresholds for basic
                sentiment = 'positive'
            elif polarity < -0.15:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': 0.0 # Cannot determine subjectivity with basic lexicon
        }
    
    def analyze_sentiment_vader(self, text):
        """VADER sentiment analysis (good for social media)"""
        if not HAS_NLTK or self.sia is None:
            # Fallback to basic analysis if VADER is not available
            return self.analyze_sentiment_basic(text)
        
        try:
            scores = self.sia.polarity_scores(text)
            
            # Determine sentiment based on compound score
            if scores['compound'] >= 0.05:
                sentiment = 'positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'compound': scores['compound'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
        except Exception as e:
            print(f"VADER analysis failed: {e}. Falling back to basic analysis.")
            return self.analyze_sentiment_basic(text)
    
    def analyze_sentiment_transformer(self, text):
        """Advanced sentiment analysis using transformers"""
        if not HAS_TRANSFORMERS or self.transformer_analyzer is None:
            # Fallback to basic analysis if transformer is not available
            basic_result = self.analyze_sentiment_basic(text)
            return {
                'sentiment': basic_result['sentiment'],
                'confidence': abs(basic_result['polarity']) # Use absolute polarity as confidence
            }
        
        try:
            result = self.transformer_analyzer(text[:512])  # Limit text length for performance
            sentiment_label = result[0]['label'].lower()
            confidence = result[0]['score']
            
            # Normalize sentiment labels from models like RoBERTa (LABEL_0, LABEL_1, LABEL_2 for neg, neu, pos)
            # Or others that might use 'positive', 'negative', 'neutral' directly
            if 'pos' in sentiment_label or 'label_2' in sentiment_label:
                sentiment = 'positive'
            elif 'neg' in sentiment_label or 'label_0' in sentiment_label:
                sentiment = 'negative'
            elif 'neu' in sentiment_label or 'label_1' in sentiment_label:
                sentiment = 'neutral'
            else:
                sentiment = 'neutral' # Default if label is unexpected
            
            return {
                'sentiment': sentiment,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Transformer analysis failed: {e}. Falling back to basic analysis.")
            basic_result = self.analyze_sentiment_basic(text)
            return {
                'sentiment': basic_result['sentiment'],
                'confidence': abs(basic_result['polarity'])
            }
    
    def extract_emotions(self, text):
        """Extract specific emotions from text using lexicon-based approach"""
        emotion_lexicon = {
            'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'fantastic', 'great', 'excellent'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'disgusted', 'annoyed', 'frustrated'],
            'sadness': ['sad', 'depressed', 'disappointed', 'upset', 'heartbroken', 'miserable'],
            'fear': ['scared', 'afraid', 'terrified', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'trust': ['trust', 'confident', 'secure', 'reliable', 'dependable'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic', 'looking forward']
        }
        
        if HAS_NLTK:
            try:
                words = word_tokenize(text.lower())
            except Exception:
                words = text.lower().split() # Fallback if NLTK tokenizer fails
        else:
            words = text.lower().split()
        
        emotion_scores = {}
        
        for emotion, keywords in emotion_lexicon.items():
            score = sum(1 for word in words if word in keywords)
            emotion_scores[emotion] = score
        
        return emotion_scores
    
    def analyze_text_batch(self, texts, method='all'):
        """Analyze sentiment for a batch of texts"""
        results = []
        
        for text in texts:
            if not text:
                continue
            if HAS_PANDAS and pd.isna(text): # Use pd.isna only if pandas is available
                continue
                
            processed_text = self.preprocess_text(text)
            result = {'original_text': text, 'processed_text': processed_text}
            
            # Basic analysis (TextBlob or custom lexicon)
            if method in ['all', 'basic']:
                basic_analysis = self.analyze_sentiment_basic(processed_text)
                # Map TextBlob/Basic keys to generic ones for consistency across methods
                result['sentiment'] = basic_analysis['sentiment']
                result['polarity'] = basic_analysis.get('polarity')
                result['subjectivity'] = basic_analysis.get('subjectivity')
                
            # VADER analysis
            if method in ['all', 'vader']:
                vader_result = self.analyze_sentiment_vader(processed_text)
                result.update({f'vader_{k}': v for k, v in vader_result.items()})
                # Override main sentiment if 'all' is chosen and vader is preferred (last method takes precedence)
                if method == 'all' and 'compound' in vader_result:
                    result['sentiment'] = vader_result['sentiment']
            
            # Transformer analysis
            if method in ['all', 'transformer']:
                transformer_result = self.analyze_sentiment_transformer(processed_text)
                result.update({f'transformer_{k}': v for k, v in transformer_result.items()})
                # Override main sentiment if 'all' is chosen and transformer is preferred (last method takes precedence)
                if method == 'all' and 'confidence' in transformer_result:
                    result['sentiment'] = transformer_result['sentiment']
            
            # Emotion extraction
            if method in ['all', 'emotions']:
                emotions = self.extract_emotions(processed_text)
                result.update({f'emotion_{k}': v for k, v in emotions.items()})
                
            results.append(result)
        
        return pd.DataFrame(results) if HAS_PANDAS else results

# Example usage (copied from main function of the original file for demonstration)
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    sample_texts = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible quality, waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Amazing customer service and fast delivery!",
        "Not worth the price, found better alternatives.",
        "Exceeded my expectations, highly recommend!",
        "Poor packaging, item arrived damaged.",
        "Great value for money, will buy again.",
        "Confusing instructions, hard to set up.",
        "Perfect! Exactly what I was looking for.",
        "The service was mediocre, neither good nor bad.",
        "I feel so angry about this situation, it's unbearable.",
        "Lost my keys, feeling a bit sad.",
        "The movie was surprisingly good, I didn't expect that!",
        "I trust this brand completely, their products are always reliable."
    ]

    print("Analyzing sample texts with 'all' methods...")
    analysis_results = analyzer.analyze_text_batch(sample_texts, method='all')

    if HAS_PANDAS:
        print("\nSentiment Analysis Results (DataFrame Head):")
        display_cols = [col for col in ['original_text', 'sentiment', 'polarity', 'vader_compound', 'transformer_confidence'] if col in analysis_results.columns]
        if not analysis_results.empty and display_cols:
            print(analysis_results[display_cols].head().to_string())
        else:
            print("No results to display or required columns are missing.")
        
        print("\nFull DataFrame Info:")
        analysis_results.info()

        print("\nSentiment Distribution:")
        print(analysis_results['sentiment'].value_counts())
        
        print("\nAverage Polarity by Sentiment:")
        print(analysis_results.groupby('sentiment')['polarity'].mean())

        print("\nTotal Emotion Counts:")
        emotion_cols = [col for col in analysis_results.columns if col.startswith('emotion_')]
        if emotion_cols:
            print(analysis_results[emotion_cols].sum())
        else:
            print("No emotion data found in results.")

    else:
        print("\nSentiment Analysis Results (List of Dictionaries - first 5):")
        for i, res in enumerate(analysis_results[:5]):
            print(f"  Original: '{res.get('original_text', 'N/A')}'")
            print(f"  Sentiment: {res.get('sentiment', 'N/A')}, Polarity: {res.get('polarity', 'N/A'):.2f}")
            print(f"  VADER Compound: {res.get('vader_compound', 'N/A'):.2f}")
            print(f"  Transformer Confidence: {res.get('transformer_confidence', 'N/A'):.2f}")
            print(f"  Emotions: {res.get('emotions', 'N/A')}\n")
