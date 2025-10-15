from flask import Flask, render_template, request, jsonify, session
import os
import json
import uuid
from datetime import datetime
from typing import Dict
from lightweight_counseling_ai import LightweightCounselingAI

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here-change-in-production')

# Initialize AI components
try:
    counseling_ai = LightweightCounselingAI()
    print("AI components initialized successfully")
except Exception as e:
    print(f"Error initializing AI components: {e}")
    print("Some features may not be available")
    counseling_ai = None

def save_session_data(user_responses: Dict, analysis: Dict) -> str:
    """Save session data directly to JSON file"""
    try:
        # Load existing data
        data_file = "anonymous_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"sessions": []}
        
        # Create session entry
        session_entry = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_responses": user_responses,
            "analysis": analysis
        }
        
        # Add to sessions
        data["sessions"].append(session_entry)
        
        # Save back to file
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return session_entry["session_id"]
        
    except Exception as e:
        print(f"Error saving session data: {e}")
        return str(uuid.uuid4())

def get_statistics() -> Dict:
    """Get anonymous statistics about counseling sessions"""
    try:
        data_file = "anonymous_data.json"
        if not os.path.exists(data_file):
            return {"total_sessions": 0, "average_emotional_score": 0}
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sessions = data.get("sessions", [])
        total_sessions = len(sessions)
        
        if total_sessions == 0:
            return {"total_sessions": 0, "average_emotional_score": 0}
        
        # Calculate average emotional score
        total_score = 0
        for session in sessions:
            analysis = session.get("analysis", {})
            emotional_score = analysis.get("emotional_score", 50)
            total_score += emotional_score
        
        average_emotional_score = total_score / total_sessions
        
        return {
            "total_sessions": total_sessions,
            "average_emotional_score": round(average_emotional_score, 2)
        }
        
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return {"total_sessions": 0, "average_emotional_score": 0}

def get_emotional_trends() -> Dict:
    """Get emotional wellness trends"""
    try:
        data_file = "anonymous_data.json"
        if not os.path.exists(data_file):
            return {"trends": []}
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sessions = data.get("sessions", [])
        if len(sessions) < 2:
            return {"trends": []}
        
        # Get recent sessions (last 10)
        recent_sessions = sessions[-10:]
        trends = []
        
        for session in recent_sessions:
            analysis = session.get("analysis", {})
            emotional_score = analysis.get("emotional_score", 50)
            timestamp = session.get("timestamp", "")
            
            trends.append({
                "timestamp": timestamp,
                "emotional_score": emotional_score
            })
        
        return {"trends": trends}
        
    except Exception as e:
        print(f"Error getting emotional trends: {e}")
        return {"trends": []}

def sanitize_input(text: str) -> str:
    """Sanitize user input for security"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    text = text.replace('<', '').replace('>', '').replace('"', '').replace("'", '')
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000]
    
    return text.strip()

def calculate_emotional_score(user_responses: Dict) -> float:
    """Calculate an emotional wellness score based on user responses"""
    score = 0.0
    max_score = 100.0
    
    # Stress level (1-10, lower is better)
    stress_level = user_responses.get('stress_level', 5)
    stress_score = (11 - stress_level) * 10  # Convert to 0-100 scale
    score += stress_score * 0.3  # 30% weight
    
    # Sleep quality (1-10, higher is better)
    sleep_quality = user_responses.get('sleep_quality', 5)
    sleep_score = sleep_quality * 10  # Convert to 0-100 scale
    score += sleep_score * 0.25  # 25% weight
    
    # Social support (1-10, higher is better)
    social_support = user_responses.get('social_support', 5)
    social_score = social_support * 10  # Convert to 0-100 scale
    score += social_score * 0.25  # 25% weight
    
    # Text sentiment analysis (20% weight)
    text_fields = [
        user_responses.get('current_mood', ''),
        user_responses.get('emotional_state', ''),
        user_responses.get('challenges', ''),
        user_responses.get('goals', '')
    ]
    
    combined_text = ' '.join([str(field) for field in text_fields if field])
    if combined_text:
        # Simple keyword-based scoring
        positive_words = ['good', 'great', 'happy', 'positive', 'confident', 'motivated', 'peaceful']
        negative_words = ['bad', 'terrible', 'sad', 'negative', 'anxious', 'depressed', 'stressed']
        
        positive_count = sum(1 for word in positive_words if word in combined_text.lower())
        negative_count = sum(1 for word in negative_words if word in combined_text.lower())
        
        if positive_count > negative_count:
            text_score = 80 + (positive_count - negative_count) * 5
        elif negative_count > positive_count:
            text_score = 20 - (negative_count - positive_count) * 5
        else:
            text_score = 50
        
        text_score = max(0, min(100, text_score))  # Clamp to 0-100
        score += text_score * 0.2
    
    return round(score, 1)

def get_risk_level(emotional_score: float, risk_factors: list) -> str:
    """Determine risk level based on emotional score and risk factors"""
    if emotional_score < 30 or len(risk_factors) >= 4:
        return "High"
    elif emotional_score < 60 or len(risk_factors) >= 2:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def index():
    """Main page with counseling form"""
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    """Handle form submission and generate AI recommendations"""
    import threading
    import time
    
    try:
        # Get form data
        form_data = request.get_json()
        
        # Generate unique user ID for session
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
        
        # Extract and sanitize user responses
        user_responses = {
            'current_mood': sanitize_input(form_data.get('current_mood', '')),
            'stress_level': int(form_data.get('stress_level', 5)),
            'sleep_quality': int(form_data.get('sleep_quality', 5)),
            'social_support': int(form_data.get('social_support', 5)),
            'life_changes': sanitize_input(form_data.get('life_changes', '')),
            'emotional_state': sanitize_input(form_data.get('emotional_state', '')),
            'coping_mechanisms': sanitize_input(form_data.get('coping_mechanisms', '')),
            'goals': sanitize_input(form_data.get('goals', '')),
            'challenges': sanitize_input(form_data.get('challenges', '')),
            'previous_counseling': sanitize_input(form_data.get('previous_counseling', '')),
            'medication': sanitize_input(form_data.get('medication', '')),
            'support_system': sanitize_input(form_data.get('support_system', ''))
        }
        
        # Compute emotional score early and include in user_responses for ML personalization
        emotional_score = calculate_emotional_score(user_responses)
        user_responses['emotional_score'] = emotional_score

        # Store in session for analysis
        session['user_responses'] = user_responses
        
        # Generate AI recommendations (dynamic ML-powered)
        if counseling_ai:
            recommendations = counseling_ai.generate_recommendations(user_responses, user_id)
        else:
            # Fallback when AI is not available
            recommendations = {
                'sections': {
                    'main': 'AI system is temporarily unavailable. Please try again later or contact support.',
                    'emotions': 'Emotional support services are currently being updated.',
                    'insights': 'Analysis features are temporarily offline.',
                    'next_steps': 'Please try again in a few moments.'
                },
                'ml_predictions': {},
                'data_insights': {'error': 'AI system unavailable'}
            }
        
        # Get analysis
        if counseling_ai:
            analysis = counseling_ai.analyze_emotional_state(user_responses, user_id)
        else:
            analysis = {'risk_factors': [], 'ml_insights': {}}
        
        # Get risk level
        risk_level = get_risk_level(emotional_score, analysis.get('risk_factors', []))
        
        # Save all form data to JSON file
        session_id = save_session_data(
            user_responses=user_responses,
            analysis=analysis
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'ml_recommendations': recommendations.get('ml_recommendations', []),
            'ml_predictions': recommendations.get('ml_predictions', {}),
            'data_insights': recommendations.get('data_insights', {}),
            'emotional_score': emotional_score,
            'analysis': analysis,
            'risk_level': risk_level,
            'user_id': user_id,
            'session_id': session_id,
            'ml_insights': analysis.get('ml_insights', {}),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in submit_form: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/analysis')
def analysis():
    """Show detailed analysis and recommendations"""
    if 'user_responses' not in session:
        return render_template('index.html')
    
    user_responses = session['user_responses']
    user_id = session.get('user_id', str(uuid.uuid4()))
    
    # Use dynamic ML-powered recommendations
    recommendations = counseling_ai.generate_recommendations(user_responses, user_id)
    ml_recommendations = recommendations.get('ml_recommendations', [])
    ml_predictions = recommendations.get('ml_predictions', {})
    model_confidence = recommendations.get('model_confidence', 0.5)
    data_insights = recommendations.get('data_insights', {})
    algorithms_used = ['Dynamic ML Engine', 'Online Learning', 'Pattern Recognition']
    
    emotional_score = calculate_emotional_score(user_responses)
    analysis = counseling_ai.analyze_emotional_state(user_responses, user_id)
    risk_level = get_risk_level(emotional_score, analysis.get('risk_factors', []))
    
    return render_template('analysis.html', 
                         user_responses=user_responses,
                         recommendations=recommendations,
                         ml_recommendations=ml_recommendations,
                         ml_predictions=ml_predictions,
                         model_confidence=model_confidence,
                         data_insights=data_insights,
                         algorithms_used=algorithms_used,
                         emotional_score=emotional_score,
                         analysis=analysis,
                         risk_level=risk_level,
                         ml_insights=analysis.get('ml_insights', {}))

@app.route('/resources')
def resources():
    """Show additional mental health resources"""
    resources = counseling_ai.get_mental_health_resources()
    return render_template('resources.html', resources=resources)

@app.route('/about')
def about():
    """About page explaining the AI system"""
    return render_template('about.html')

@app.route('/api/health_check')
def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'ai_system': counseling_ai is not None,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Download required NLTK data if available
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("[CHECK] NLTK data downloaded successfully")
    except Exception as e:
        print(f"NLTK not available: {e}")
        print("System will use basic text processing")
    
    # Create necessary directories
    os.makedirs('ml_models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
@app.route('/data/statistics')
def statistics_route():
    """Get anonymous statistics about counseling sessions"""
    try:
        stats = get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data/trends')
def get_trends():
    """Get emotional wellness trends"""
    try:
        trends = get_emotional_trends()
        return jsonify({
            'success': True,
            'trends': trends,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_system': 'Lightweight Dynamic ML Counseling AI',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("AI Counseling System - Lightweight Dynamic ML Version")
    print("=" * 60)
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
