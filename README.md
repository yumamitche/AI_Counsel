# Pure Python AI Counseling System

A dynamic, laptop-optimized AI counseling system that provides personalized mental health recommendations using pure Python machine learning with online learning capabilities - no external ML libraries required!

## 🚀 Quick Start

### Option 1: Run with lightweight launcher
```bash
python run_lightweight.py
```

### Option 2: Direct run
```bash
python app.py
```

### Option 3: Direct Flask run
```bash
flask run
```

Then open your browser and go to `http://localhost:5000`

## 🧠 Features

### Pure Python Machine Learning
- **Online Learning**: Models update in real-time with new data
- **Pure Python ML**: Custom implementations - no external ML libraries required
- **Pattern Recognition**: K-Means clustering, Decision Trees, Linear Regression
- **Risk Assessment**: Automatic crisis detection and intervention
- **User Clustering**: Groups similar users for better recommendations

### Dynamic Recommendation System
- **No Hardcoded Templates**: All recommendations generated algorithmically
- **Pattern Recognition**: Analyzes user text to match with appropriate techniques
- **Collaborative Filtering**: Uses similar users' successful interventions
- **Real-Time Learning**: Learns from historical user data and patterns
- **Personalized Content**: Based on ML analysis and user characteristics

### Lightweight NLP
- **Sentiment Analysis**: Real-time emotional state assessment (NLTK optional)
- **Text Processing**: Basic keyword and pattern recognition
- **Context Awareness**: Understanding of user context and history
- **Dynamic Responses**: All content generated dynamically

### Laptop-Optimized Performance
- **Minimal Dependencies**: Only essential libraries (numpy, pandas) - no scikit-learn required
- **Memory Efficient**: Optimized for laptop hardware
- **Fast Training**: Pure Python algorithms for quick model updates
- **Incremental Learning**: Models learn from each new session

## 📁 File Structure

```
ai_counsel/
├── app.py                          # Main Flask application
├── lightweight_counseling_ai.py    # Lightweight AI system
├── pure_python_ml_engine.py        # Pure Python ML engine
├── dynamic_recommendation_engine.py # Dynamic recommendation system
├── run_lightweight.py              # Optimized launcher
├── templates/                      # HTML templates
│   ├── index.html
│   ├── analysis.html
│   ├── resources.html
│   └── about.html
├── ml_models/                      # ML model storage (auto-created)
│   ├── user_profiles.json         # User clustering data
│   └── dynamic_knowledge.json     # ML knowledge base
├── requirements.txt                # Lightweight dependencies
├── DYNAMIC_RECOMMENDATIONS_GUIDE.md # Dynamic system guide
└── README.md                       # This file
```

## 🔧 Requirements

### Core Dependencies (Minimal)
- Python 3.8+
- Flask 2.3.3
- numpy 1.24.3
- pandas 2.0.3

### Optional Dependencies
- NLTK 3.8.1 (for advanced NLP - system works without it)

### Installation
```bash
pip install -r requirements.txt
```

## 📊 How It Works

1. **User Input**: User fills out counseling form
2. **Pure Python ML Analysis**: AI analyzes using custom ML models
3. **Online Learning**: Models update with new session data
4. **Personalized Recommendations**: Generates ML-powered suggestions
5. **Continuous Improvement**: System learns and adapts in real-time

## 🎯 ML Algorithms Used

### Core Models (Pure Python)
- **K-Means Clustering**: User segmentation and pattern recognition
- **Decision Tree**: Risk classification using Gini impurity
- **Linear Regression**: Success prediction with gradient descent
- **Custom Algorithms**: All implemented in pure Python

### Features
- **Risk Classification**: High/Medium/Low risk assessment
- **Success Prediction**: Intervention success probability
- **User Clustering**: Groups similar users for recommendations
- **Online Learning**: Models update with each new session

## 🛡️ Privacy & Security

- All data is anonymized and stored locally
- No external API calls or data transmission
- Local processing only
- Secure session management
- Crisis detection and escalation

## ⚡ Performance Optimizations

### Laptop-Friendly Features
- **Pure Python Implementation**: No external ML library compilation issues
- **Single-threaded Processing**: Optimized for laptop CPUs
- **Memory Efficient**: Minimal memory footprint
- **Fast Startup**: Quick model initialization
- **Incremental Updates**: Only retrain when necessary

### Learning Efficiency
- **Batch Learning**: Updates every 5 new sessions
- **Adaptive Learning Rate**: Automatically adjusts learning speed
- **Model Persistence**: Saves trained models for quick startup
- **Fallback Systems**: Works even without ML libraries

## ⚠️ Important Note

This AI system provides supportive recommendations and is **NOT a substitute for professional medical advice**. Always seek qualified health providers for mental health conditions.

## 🆘 Crisis Resources

If you or someone you know is in crisis:
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **Emergency Services**: 911

## 🎯 Usage

1. **Start the system**: Run `python run_lightweight.py`
2. **Open browser**: Go to `http://localhost:5000`
3. **Fill out form**: Complete the counseling assessment
4. **Get ML recommendations**: Receive dynamic AI-generated advice
5. **System learns**: Models improve with each session

The system will continuously learn and adapt to provide better recommendations over time using online learning algorithms.

## 🔄 Dynamic Learning

### How Online Learning Works
- **Session Collection**: Each user session is added to the training data
- **Incremental Updates**: Models update every 5 new sessions
- **Pattern Recognition**: Identifies trends and user clusters
- **Adaptive Recommendations**: Suggestions improve based on learned patterns
- **Performance Tracking**: Monitors model accuracy and effectiveness

### Learning Metrics
- **Total Sessions**: Number of counseling sessions processed
- **Model Accuracy**: Performance of ML algorithms
- **User Clusters**: Number of identified user groups
- **Success Rate**: Intervention effectiveness predictions

---

**Built with ❤️ using pure Python machine learning**