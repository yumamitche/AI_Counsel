#!/usr/bin/env python3
"""
Lightweight Counseling AI - Optimized for laptop performance
Uses dynamic ML engine with minimal dependencies
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the pure Python ML engine
try:
    from pure_python_ml_engine import PurePythonMLEngine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False

# Lightweight NLP (optional)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class LightweightCounselingAI:
    """Lightweight counseling AI optimized for laptop performance"""
    
    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize pure Python ML engine
        if ML_ENGINE_AVAILABLE:
            self.ml_engine = PurePythonMLEngine()
            print("✅ Pure Python ML Engine initialized")
        else:
            self.ml_engine = None
            print("⚠️ Pure Python ML Engine not available - using rule-based fallback")
        
        # Initialize lightweight NLP
        if NLTK_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                print("✅ Lightweight NLP initialized")
            except:
                self.sentiment_analyzer = None
                print("⚠️ NLTK sentiment analysis not available")
        else:
            self.sentiment_analyzer = None
            print("⚠️ NLTK not available - using basic text processing")
        
        # User profiles and learning data
        self.user_profiles = {}
        self.conversation_memory = {}
        
        # Load existing data
        self._load_user_profiles()
    
    def _load_user_profiles(self):
        """Load user profiles from file"""
        profiles_path = os.path.join(self.model_dir, 'user_profiles.json')
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'r') as f:
                    self.user_profiles = json.load(f)
                print(f"✅ Loaded {len(self.user_profiles)} user profiles")
            except Exception as e:
                print(f"⚠️ Error loading user profiles: {e}")
                self.user_profiles = {}
    
    def _save_user_profiles(self):
        """Save user profiles to file"""
        profiles_path = os.path.join(self.model_dir, 'user_profiles.json')
        try:
            with open(profiles_path, 'w') as f:
                json.dump(self.user_profiles, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving user profiles: {e}")
    
    def analyze_emotional_state(self, user_responses: Dict, user_id: str = None) -> Dict:
        """Analyze emotional state using lightweight methods"""
        analysis = {
            'overall_sentiment': 'neutral',
            'emotional_indicators': [],
            'risk_factors': [],
            'strengths': [],
            'confidence_score': 0.0,
            'ml_insights': {}
        }
        
        # Combine text inputs for analysis
        text_fields = [
            user_responses.get('current_mood', ''),
            user_responses.get('emotional_state', ''),
            user_responses.get('challenges', ''),
            user_responses.get('goals', '')
        ]
        
        combined_text = ' '.join([str(field) for field in text_fields if field])
        
        # Lightweight sentiment analysis
        if combined_text and self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(combined_text)
                analysis['sentiment_scores'] = sentiment_scores
                
                if sentiment_scores['compound'] > 0.1:
                    analysis['overall_sentiment'] = 'positive'
                elif sentiment_scores['compound'] < -0.1:
                    analysis['overall_sentiment'] = 'negative'
                else:
                    analysis['overall_sentiment'] = 'neutral'
            except:
                analysis['overall_sentiment'] = self._basic_sentiment_analysis(combined_text)
        else:
            analysis['overall_sentiment'] = self._basic_sentiment_analysis(combined_text)
        
        # Risk assessment
        analysis['risk_factors'] = self._assess_risk_factors(user_responses, combined_text)
        
        # ML insights if available
        if self.ml_engine:
            analysis['ml_insights'] = {
                'ml_available': True,
                'total_sessions': len(self.ml_engine.session_data),
                'models_active': len(self.ml_engine.models)
            }
        else:
            analysis['ml_insights'] = {'ml_available': False}
        
        # Update user profile
        if user_id:
            self._update_user_profile(user_id, analysis, user_responses)
        
        return analysis
    
    def _basic_sentiment_analysis(self, text: str) -> str:
        """Basic sentiment analysis without external libraries"""
        if not text:
            return 'neutral'
        
        positive_words = ['good', 'great', 'happy', 'positive', 'confident', 'motivated', 'peaceful', 'excited', 'pleased', 'wonderful']
        negative_words = ['bad', 'terrible', 'sad', 'negative', 'anxious', 'depressed', 'stressed', 'worried', 'angry', 'awful']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_risk_factors(self, user_responses: Dict, text: str) -> List[str]:
        """Assess risk factors using lightweight methods"""
        risk_factors = []
        
        # Text-based risk assessment
        text_lower = text.lower()
        risk_patterns = {
            'suicidal_ideation': ['kill myself', 'end it all', 'not worth living', 'suicide'],
            'self_harm': ['hurt myself', 'cut myself', 'self harm', 'self injury'],
            'severe_depression': ['hopeless', 'worthless', 'empty', 'numb', 'dead inside'],
            'severe_anxiety': ['panic attack', 'can\'t breathe', 'losing control', 'going crazy']
        }
        
        for risk_type, keywords in risk_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                risk_factors.append(risk_type.replace('_', ' ').title())
        
        # Numeric risk factors
        if user_responses.get('stress_level', 5) >= 9:
            risk_factors.append('Extreme Stress')
        if user_responses.get('sleep_quality', 5) <= 2:
            risk_factors.append('Severe Sleep Deprivation')
        if user_responses.get('social_support', 5) <= 2:
            risk_factors.append('Social Isolation')
        
        return risk_factors
    
    def _update_user_profile(self, user_id: str, analysis: Dict, user_responses: Dict):
        """Update user profile with new data"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'created': datetime.now().isoformat(),
                'sessions': [],
                'baselines': {},
                'patterns': {}
            }
        
        # Add session data
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'responses': user_responses,
            'session_id': f"session_{len(self.user_profiles[user_id]['sessions']) + 1}"
        }
        self.user_profiles[user_id]['sessions'].append(session_data)
        
        # Update baselines
        self._update_baselines(user_id, user_responses)
        
        # Save profiles
        self._save_user_profiles()
    
    def _update_baselines(self, user_id: str, user_responses: Dict):
        """Update user baselines"""
        sessions = self.user_profiles[user_id]['sessions']
        
        if len(sessions) >= 3:  # Need minimum data for baselines
            stress_levels = [s['responses'].get('stress_level', 5) for s in sessions]
            sleep_qualities = [s['responses'].get('sleep_quality', 5) for s in sessions]
            social_supports = [s['responses'].get('social_support', 5) for s in sessions]
            
            self.user_profiles[user_id]['baselines'] = {
                'avg_stress': sum(stress_levels) / len(stress_levels),
                'avg_sleep': sum(sleep_qualities) / len(sleep_qualities),
                'avg_social': sum(social_supports) / len(social_supports),
                'session_count': len(sessions)
            }
    
    def generate_recommendations(self, user_responses: Dict, user_id: str = None) -> Dict:
        """Generate recommendations using dynamic ML if available"""
        # Get ML-powered recommendations if available
        if self.ml_engine:
            try:
                # Add session to ML engine for learning
                session_data = {
                    'user_id': user_id or 'anonymous',
                    'timestamp': datetime.now().isoformat(),
                    **user_responses
                }
                self.ml_engine.add_session(session_data)
                
                # Get ML recommendations
                ml_results = self.ml_engine.get_dynamic_recommendations(user_responses)
                
                # Combine with basic recommendations
                basic_recommendations = self._generate_basic_recommendations(user_responses, user_id)
                
                return {
                    **basic_recommendations,
                    'ml_predictions': ml_results['ml_predictions'],
                    'ml_recommendations': ml_results['ml_recommendations'],
                    'sections': ml_results.get('sections', {}),
                    'model_confidence': ml_results['model_confidence'],
                    'data_insights': ml_results['data_insights'],
                    'recommendation_source': 'Pure Python ML Engine',
                    'personalization_level': 'ml_powered'
                }
            except Exception as e:
                print(f"⚠️ ML engine error: {e}")
                return self._generate_basic_recommendations(user_responses, user_id)
        else:
            return self._generate_basic_recommendations(user_responses, user_id)
    
    def _generate_basic_recommendations(self, user_responses: Dict, user_id: str = None) -> Dict:
        """Generate basic recommendations without ML"""
        recommendations = {
            'immediate_actions': self._generate_immediate_actions(user_responses),
            'therapeutic_techniques': self._generate_therapeutic_techniques(user_responses),
            'lifestyle_changes': self._generate_lifestyle_changes(user_responses),
            'professional_help': self._generate_professional_help(user_responses),
            'self_help_resources': self._generate_self_help_resources(user_responses),
            'crisis_resources': self._generate_crisis_resources(user_responses),
            'follow_up_plan': self._generate_follow_up_plan(user_responses, user_id),
            'recommendation_source': 'Rule-based System',
            'personalization_level': 'basic'
        }
        
        return recommendations
    
    def _generate_immediate_actions(self, user_responses: Dict) -> List[str]:
        """Generate immediate actions"""
        actions = []
        
        stress_level = user_responses.get('stress_level', 5)
        sleep_quality = user_responses.get('sleep_quality', 5)
        social_support = user_responses.get('social_support', 5)
        
        # Crisis intervention
        if stress_level >= 9 or sleep_quality <= 2 or social_support <= 2:
            actions.append("🚨 CRISIS ALERT: Please contact emergency services (911) or a crisis hotline immediately")
            actions.append("You are not alone - reach out to someone you trust right now")
        
        # Stress management
        if stress_level >= 7:
            actions.append("Practice the 4-7-8 breathing technique immediately")
            actions.append("Take a 10-minute break in a quiet space")
            actions.append("Use grounding technique: Name 5 things you can see, 4 you can touch, 3 you can hear")
        
        # Sleep improvement
        if sleep_quality <= 4:
            actions.append("Avoid screens 1 hour before bedtime")
            actions.append("Create a relaxing bedtime routine")
            actions.append("Keep bedroom cool, dark, and quiet")
        
        # Social connection
        if social_support <= 4:
            actions.append("Reach out to one person in your support network today")
            actions.append("Consider joining a support group or community activity")
        
        return actions
    
    def _generate_therapeutic_techniques(self, user_responses: Dict) -> List[Dict]:
        """Generate therapeutic techniques"""
        techniques = []
        
        stress_level = user_responses.get('stress_level', 5)
        current_mood = user_responses.get('current_mood', '').lower()
        
        # CBT techniques
        if 'anxiety' in current_mood or stress_level >= 6:
            techniques.append({
                'name': 'Cognitive Restructuring',
                'description': 'Challenge negative thoughts with evidence',
                'steps': [
                    'Identify the automatic thought',
                    'Rate the emotion intensity (1-10)',
                    'Challenge the thought with evidence',
                    'Generate alternative balanced thoughts',
                    'Re-rate emotion intensity'
                ]
            })
        
        # Mindfulness techniques
        techniques.append({
            'name': 'Mindfulness Meditation',
            'description': 'Present-moment awareness practice',
            'steps': [
                'Find a comfortable position',
                'Focus on your breath naturally',
                'Notice thoughts without judgment',
                'Return to breath when distracted',
                'End with gratitude reflection'
            ]
        })
        
        # Behavioral activation
        if 'depression' in current_mood or user_responses.get('sleep_quality', 5) <= 4:
            techniques.append({
                'name': 'Behavioral Activation',
                'description': 'Increase positive activities to improve mood',
                'steps': [
                    'List 5 activities you used to enjoy',
                    'Choose 1 activity to do today',
                    'Start with small, achievable steps',
                    'Track your mood before and after',
                    'Gradually increase activity level'
                ]
            })
        
        return techniques
    
    def _generate_lifestyle_changes(self, user_responses: Dict) -> List[str]:
        """Generate lifestyle recommendations"""
        changes = []
        
        stress_level = user_responses.get('stress_level', 5)
        sleep_quality = user_responses.get('sleep_quality', 5)
        social_support = user_responses.get('social_support', 5)
        
        if stress_level >= 6:
            changes.append("Implement daily stress management: 10 minutes meditation or deep breathing")
            changes.append("Regular physical exercise: 30 minutes, 3 times per week")
            changes.append("Limit caffeine and alcohol intake")
        
        if sleep_quality <= 5:
            changes.append("Maintain consistent sleep schedule (same bedtime and wake time)")
            changes.append("Create relaxing bedtime routine")
            changes.append("Avoid screens 1 hour before bed")
        
        if social_support <= 5:
            changes.append("Schedule regular social activities")
            changes.append("Join clubs or groups with similar interests")
            changes.append("Practice active listening in conversations")
        
        # General wellness
        changes.append("Eat balanced meals with regular timing")
        changes.append("Stay hydrated throughout the day")
        changes.append("Spend time in nature when possible")
        
        return changes
    
    def _generate_professional_help(self, user_responses: Dict) -> List[str]:
        """Generate professional help recommendations"""
        recommendations = []
        
        stress_level = user_responses.get('stress_level', 5)
        sleep_quality = user_responses.get('sleep_quality', 5)
        social_support = user_responses.get('social_support', 5)
        
        # High risk indicators
        if stress_level >= 8 or sleep_quality <= 3 or social_support <= 3:
            recommendations.append("🔴 HIGH PRIORITY: Schedule immediate consultation with a mental health professional")
            recommendations.append("Consider intensive outpatient program based on your risk profile")
        
        # Moderate risk
        elif stress_level >= 6 or sleep_quality <= 5 or social_support <= 5:
            recommendations.append("🟡 MODERATE PRIORITY: Consider scheduling with a therapist or counselor")
            recommendations.append("Look for therapists specializing in your specific concerns")
        
        # General recommendations
        recommendations.append("Seek therapist specializing in evidence-based treatments (CBT, DBT)")
        recommendations.append("Consider group therapy for additional support")
        recommendations.append("Ask your primary care doctor for mental health referrals")
        
        return recommendations
    
    def _generate_self_help_resources(self, user_responses: Dict) -> List[Dict]:
        """Generate self-help resources"""
        resources = []
        
        current_mood = user_responses.get('current_mood', '').lower()
        
        # Mood-specific resources
        if 'anxiety' in current_mood:
            resources.append({
                'type': 'App',
                'name': 'Headspace',
                'description': 'Meditation and mindfulness for anxiety',
                'url': 'https://www.headspace.com'
            })
            resources.append({
                'type': 'Book',
                'name': 'The Anxiety and Worry Workbook',
                'description': 'Evidence-based CBT techniques for anxiety',
                'author': 'David A. Clark'
            })
        
        if 'depression' in current_mood:
            resources.append({
                'type': 'App',
                'name': 'MoodTools',
                'description': 'Depression self-help with mood tracking',
                'url': 'https://moodtools.org'
            })
        
        # General resources
        resources.append({
            'type': 'Website',
            'name': 'Mental Health America',
            'description': 'Comprehensive mental health resources',
            'url': 'https://www.mhanational.org'
        })
        
        resources.append({
            'type': 'App',
            'name': 'Calm',
            'description': 'Sleep and relaxation app',
            'url': 'https://www.calm.com'
        })
        
        return resources
    
    def _generate_crisis_resources(self, user_responses: Dict) -> List[Dict]:
        """Generate crisis resource information"""
        resources = [
            {
                'name': 'National Suicide Prevention Lifeline',
                'phone': '988',
                'description': '24/7 crisis support',
                'url': 'https://988lifeline.org'
            },
            {
                'name': 'Crisis Text Line',
                'text': 'HOME to 741741',
                'description': 'Text-based crisis support',
                'url': 'https://www.crisistextline.org'
            },
            {
                'name': 'Emergency Services',
                'phone': '911',
                'description': 'For immediate life-threatening situations'
            }
        ]
        
        return resources
    
    def _generate_follow_up_plan(self, user_responses: Dict, user_id: str = None) -> Dict:
        """Generate follow-up plan"""
        plan = {
            'daily_practices': [
                "Morning mindfulness (5 minutes)",
                "Evening reflection (5 minutes)",
                "Gratitude journaling"
            ],
            'weekly_check_ins': [
                "Mood and stress assessment",
                "Review and adjust coping strategies",
                "Track progress on goals"
            ],
            'monthly_reviews': [
                "Progress evaluation",
                "Strategy effectiveness review",
                "Plan modifications as needed"
            ],
            'progress_tracking': [
                "Daily mood tracking",
                "Weekly stress level assessment",
                "Monthly wellness review"
            ]
        }
        
        # Personalize based on user history
        if user_id and user_id in self.user_profiles:
            session_count = len(self.user_profiles[user_id]['sessions'])
            if session_count >= 5:
                plan['personalization_level'] = 'advanced'
                plan['adaptive_recommendations'] = True
            else:
                plan['personalization_level'] = 'basic'
        
        return plan
    
    def get_mental_health_resources(self) -> Dict:
        """Get comprehensive mental health resources"""
        return {
            'hotlines': [
                {
                    'name': 'National Suicide Prevention Lifeline',
                    'phone': '988',
                    'description': '24/7 crisis support'
                },
                {
                    'name': 'Crisis Text Line',
                    'text': 'HOME to 741741',
                    'description': 'Text-based crisis support'
                }
            ],
            'organizations': [
                {
                    'name': 'Mental Health America',
                    'url': 'https://www.mhanational.org',
                    'description': 'Support, education, and advocacy'
                },
                {
                    'name': 'National Alliance on Mental Illness (NAMI)',
                    'url': 'https://www.nami.org',
                    'description': 'Support groups and education'
                }
            ],
            'apps': [
                {
                    'name': 'Headspace',
                    'description': 'Meditation and mindfulness',
                    'url': 'https://www.headspace.com'
                },
                {
                    'name': 'Calm',
                    'description': 'Sleep and relaxation',
                    'url': 'https://www.calm.com'
                }
            ]
        }
    
    def get_system_status(self) -> Dict:
        """Get system status and performance"""
        status = {
            'ml_engine_available': ML_ENGINE_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE,
            'total_users': len(self.user_profiles),
            'total_sessions': sum(len(profile['sessions']) for profile in self.user_profiles.values()),
            'last_updated': datetime.now().isoformat()
        }
        
        if self.ml_engine:
            status['ml_performance'] = self.ml_engine.get_model_performance()
        
        return status

# Backward compatibility
DynamicCounselingAI = LightweightCounselingAI
CounselingAI = LightweightCounselingAI
