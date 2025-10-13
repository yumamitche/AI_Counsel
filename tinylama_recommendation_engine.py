#!/usr/bin/env python3
"""
TinyLlama Recommendation Engine - Dynamic Text Generation
Generates personalized counseling recommendations using TinyLlama language model
"""

import json
from typing import Dict, List, Optional
import os
from datetime import datetime
import re

# TinyLlama and transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    # Test if PyTorch is working properly
    torch.cuda.is_available()  # This will fail if PyTorch has issues
    TINYLLAMA_AVAILABLE = True
except (ImportError, Exception) as e:
    TINYLLAMA_AVAILABLE = False
    print(f"⚠️ Transformers/PyTorch not available: {e}")
    print("⚠️ Using fallback recommendations")

class TinyLlamaRecommendationEngine:
    """Dynamic recommendation engine using TinyLlama-Chat for text generation"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # Allow overriding the model location with environment variable
        # If LOCAL_TINYLLAMA_PATH is set, we'll use that local directory and prevent downloads
        local_path = os.environ.get('LOCAL_TINYLLAMA_PATH')
        # Auto-detect common local model folder if env var not set
        if not local_path:
            candidate = os.path.join(os.path.dirname(__file__), 'ml_models', 'tinyllama')
            if os.path.isdir(candidate):
                local_path = candidate

        if local_path:
            # If a local path is provided or detected, prefer it as the model name
            self.model_name = local_path
            self._local_files_only = True
        else:
            self.model_name = model_name
            self._local_files_only = False
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.device = "cpu"  # Force CPU for laptop compatibility
        
        # Model configuration for laptop performance (TinyLlama-Chat optimized)
        self.model_config = {
            'max_new_tokens': 800,  # Very high limit for extremely comprehensive responses
            'min_new_tokens': 300,  # High minimum for very detailed content
            'temperature': 0.7,     # Balanced for quality and speed
            'top_p': 0.85,          # Balanced for quality and speed
            'top_k': 40,            # Balanced for quality and speed
            'do_sample': True,
            'no_repeat_ngram_size': 3,  # Balanced for quality and speed
            'repetition_penalty': 1.1,  # Balanced for quality and speed
            'pad_token_id': None,
            'eos_token_id': None
        }
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize TinyLlama-Chat model for text generation"""
        if not TINYLLAMA_AVAILABLE:
            print("⚠️ Transformers not available - using fallback recommendations")
            return
        
        try:
            if getattr(self, '_local_files_only', False):
                print(f"🚀 Loading TinyLlama-Chat from local path: {self.model_name} (no downloads)")
            else:
                print("🚀 Loading TinyLlama-Chat (1.1B) for dynamic recommendations...")
                print("📥 This may take a moment to download the model (~2GB)...")
            
            # Load tokenizer with timeout
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left',
                local_files_only=getattr(self, '_local_files_only', False),
                use_fast=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                local_files_only=getattr(self, '_local_files_only', False)
            )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                torch_dtype=torch.float32
            )
            
            print("✅ TinyLlama-Chat model loaded successfully for dynamic recommendations")
            
        except Exception as e:
            print(f"⚠️ Failed to load TinyLlama-Chat: {e}")
            print("🔄 Falling back to template-based recommendations")
            self.generator = None
    
    def generate_counseling_recommendations(self, user_data: Dict, ml_insights: Dict) -> List[str]:
        """Generate personalized counseling recommendations using TinyLlama-Chat"""
        if not self.generator:
            return self._get_fallback_recommendations(user_data, ml_insights)
        
        try:
            # Create comprehensive prompt
            prompt = self._create_counseling_prompt(user_data, ml_insights)
            
            # Generate recommendations
            generated_text = self._generate_text(prompt)
            
            # Parse and format recommendations
            recommendations = self._parse_recommendations(generated_text)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ TinyLlama-Chat generation failed: {e}")
            return self._get_fallback_recommendations(user_data, ml_insights)

    def generate_section_paragraph(self, section: str, user_data: Dict, ml_insights: Dict) -> str:
        """Generate a single tailored paragraph for a specific UI section.

        section: one of ['main', 'emotions', 'insights', 'next_steps']
        """
        if not self.generator:
            return self._get_fallback_paragraph(section, user_data)
        
        try:
            prompt = self._create_section_prompt(section, user_data, ml_insights)
            # Try up to 2 times with validation
            for attempt in range(2):
                try:
                    text = self._generate_text(prompt)
                    text = self._postprocess_paragraph(text)
                    if self._is_valid_paragraph(text, user_data):
                        return text
                except Exception as gen_error:
                    print(f"⚠️ Generation attempt {attempt + 1} failed: {gen_error}")
                    if attempt == 1:  # Last attempt
                        return self._get_fallback_paragraph(section, user_data)
            return self._get_fallback_paragraph(section, user_data)
        except Exception as e:
            print(f"⚠️ TinyLlama-Chat section generation failed ({section}): {e}")
            return self._get_fallback_paragraph(section, user_data)

    def _create_section_prompt(self, section: str, user_data: Dict, ml: Dict) -> str:
        # Extract common inputs
        current_mood = user_data.get('current_mood', '')
        stress_level = user_data.get('stress_level', 5)
        sleep_quality = user_data.get('sleep_quality', 5)
        social_support = user_data.get('social_support', 5)
        emotional_state = user_data.get('emotional_state', '')
        challenges = user_data.get('challenges', '')
        goals = user_data.get('goals', '')
        life_changes = user_data.get('life_changes', '')
        coping_mechanisms = user_data.get('coping_mechanisms', '')

        risk_level = ml.get('risk_level', 'medium')
        text_emotions = ml.get('text_emotions', {})
        text_sentiment = ml.get('text_sentiment', {})
        success_probability = ml.get('success_probability', 0.7)

        base_header = (
            f"USER CONTEXT:\n"
            f"Current Mood: {current_mood}\n"
            f"Stress: {stress_level}/10, Sleep: {sleep_quality}/10, Social: {social_support}/10\n"
            f"Emotional State: {emotional_state}\n"
            f"Challenges: {challenges}\n"
            f"Goals: {goals}\n"
            f"Life Changes: {life_changes}\n"
            f"Coping: {coping_mechanisms}\n"
        )

        if section == 'main':
            instructions = (
                "Write ONE extremely detailed, comprehensive paragraph (30–50 sentences) providing COUNSELOR GUIDANCE for real-life counseling advice that thoroughly addresses EVERY aspect of the client's specific situation. "
                "You are providing guidance to a counselor on how to give insights and advice to their client. Consider ALL of the following client inputs: "
                f"Current Mood: '{current_mood}' - guide the counselor on strategies for this mood state; "
                f"Stress Level: {stress_level}/10 - guide the counselor on advice tailored to this specific stress level; "
                f"Sleep Quality: {sleep_quality}/10 - guide the counselor on sleep improvement strategies; "
                f"Social Support: {social_support}/10 - guide the counselor on social connection guidance; "
                f"Emotional State: '{emotional_state}' - guide the counselor on emotional support approaches; "
                f"Challenges: '{challenges}' - guide the counselor on addressing each challenge mentioned; "
                f"Goals: '{goals}' - guide the counselor on goal achievement strategies; "
                f"Life Changes: '{life_changes}' - guide the counselor on transition support; "
                f"Coping Mechanisms: '{coping_mechanisms}' - guide the counselor on building upon existing strategies. "
                "EXPAND extensively on each area. Include: 1) Detailed analysis guidance for the counselor on the client's current situation, "
                "2) Specific counseling strategies for the client's exact challenges, 3) Step-by-step approaches the counselor can use for the client's goals, "
                "4) Realistic daily/weekly routine guidance tailored to the client's stress and sleep levels, 5) Social support strategies the counselor can recommend, "
                "6) Emotional regulation techniques the counselor can teach for the client's emotional state, 7) Life transition support guidance for the counselor, "
                "8) Building upon the client's existing coping mechanisms, 9) Multiple alternative approaches the counselor can offer, "
                "10) Detailed explanations the counselor can provide on why strategies work, 11) Potential obstacles and how the counselor can help overcome them, "
                "12) Building sustainable habits guidance for the counselor, 13) Mood-specific strategies the counselor can use, 14) Stress-level appropriate techniques, "
                "15) Sleep improvement plans the counselor can recommend, 16) Social connection building guidance, 17) Goal achievement timelines for the counselor, "
                "18) Life change adaptation strategies. Focus on providing COUNSELOR GUIDANCE and INSIGHTS for effective counseling. "
                "STRICT SCOPE: counselor guidance and counseling insights only. Forbid news, articles, books, social media, marketing, technology tips, math/units, or unrelated content. "
                "No lists, no URLs, professional counseling tone, no diagnoses, no meta commentary."
            )
            title = "AI-GENERATED RECOMMENDATIONS"
        elif section == 'emotions':
            instructions = (
                "Write ONE extremely comprehensive paragraph (20–35 sentences) providing COUNSELOR GUIDANCE on how to guide the client in their emotional processing and regulation. "
                f"Consider the client's current mood '{current_mood}', stress level {stress_level}/10, sleep quality {sleep_quality}/10, social support {social_support}/10, "
                f"emotional state '{emotional_state}', challenges '{challenges}', and life changes '{life_changes}' when providing counselor guidance. "
                "EXPAND extensively on how the counselor can guide the client through emotional wellness principles. Include: 1) How the counselor can help recognize and validate the client's emotional experience, "
                "2) Emotional regulation techniques the counselor can teach (breathing, mindfulness, grounding) appropriate for the client's stress level, 3) How the counselor can guide emotional processing and expression, "
                "4) Building emotional resilience and self-awareness guidance for the counselor considering the client's challenges, 5) How the counselor can help establish healthy emotional boundaries and self-care, "
                "6) Long-term emotional wellness practices the counselor can introduce, 7) Self-compassion and emotional self-care guidance for the counselor, 8) Emotional intelligence development strategies for the counselor, "
                "9) Stress management techniques the counselor can teach for the client's specific stress level, 10) How the counselor can help the client understand emotional patterns, "
                "11) Building emotional vocabulary guidance for the counselor, 12) Creating emotional safety in the counseling relationship, 13) Developing emotional flexibility strategies for the counselor, "
                "14) Sleep-related emotional regulation guidance, 15) Social support for emotional wellbeing strategies, 16) Emotional strategies for life transitions guidance, "
                "17) Mood-specific emotional techniques for the counselor, 18) Building emotional resilience through challenges guidance. "
                "Focus on providing COUNSELOR GUIDANCE on how to effectively guide the client's emotional journey. "
                "STRICT SCOPE: counselor guidance for emotional support only; forbid news, articles, social media, marketing, or unrelated ideas. "
                "No lists, no URLs, professional counseling tone."
            )
            title = f"EMOTIONAL WELLBEING GUIDANCE (risk: {risk_level}, emotions: {text_emotions})"
        elif section == 'insights':
            instructions = (
                "Write ONE paragraph (4–7 sentences) that summarizes likely outcomes and possible results in everyday counseling language. "
                "Explain observed patterns simply, what improvement might look like for this user, and how to track progress (habits/journaling/reflection). "
                "STRICT SCOPE: counseling-only insights; forbid references to articles, books, social media, or unrelated topics. Be actionable and optimistic; no lists, no URLs."
            )
            title = f"INSIGHTS SUMMARY (sentiment: {text_sentiment}, success: {success_probability:.0%})"
        else:  # next_steps
            instructions = (
                "Write ONE extremely detailed, comprehensive paragraph (20–35 sentences) providing COUNSELOR GUIDANCE on what specific steps the client should take, serving as a guideline for the counselor. "
                f"Consider the client's current mood '{current_mood}', stress level {stress_level}/10, sleep quality {sleep_quality}/10, social support {social_support}/10, "
                f"emotional state '{emotional_state}', challenges '{challenges}', goals '{goals}', life changes '{life_changes}', and coping mechanisms '{coping_mechanisms}'. "
                "EXPAND extensively on providing counselor guidance for client action steps. Include: 1) Immediate coping activities the counselor can assign for today/this week, "
                "2) Specific activities the counselor can recommend that address the client's exact challenges and stress level, 3) Practical coping mechanisms the counselor can help build upon their current strategies, "
                "4) Activities the counselor can suggest that support the client's specific goals and emotional state, 5) Realistic daily/weekly action plans the counselor can create tailored to the client's sleep and social support levels, "
                "6) One concrete accountability step the counselor can help the client commit to, 7) Progressive coping strategies the counselor can teach for different stress levels, 8) Long-term coping skill development guidance for the counselor, "
                "9) Specific techniques the counselor can use for the client's unique challenges, 10) Detailed step-by-step instructions the counselor can provide for each activity, "
                "11) How the counselor can help the client adapt activities based on their energy and mood levels, 12) Building a comprehensive coping toolkit guidance for the counselor, "
                "13) Creating sustainable routines guidance that work with the client's lifestyle, 14) Measuring progress and adjusting strategies guidance for the counselor, "
                "15) Sleep improvement activities the counselor can recommend, 16) Social connection activities the counselor can suggest based on the client's support level, "
                "17) Goal-specific action steps the counselor can help create, 18) Life transition coping activities guidance, 19) Mood-specific coping strategies for the counselor, "
                "20) Building upon the client's existing coping mechanisms guidance. Focus on providing COUNSELOR GUIDANCE on what steps the client should take. "
                "STRICT SCOPE: counselor guidance for client action steps only; forbid references to articles, books, social media, marketing, math/units, or unrelated content. No lists, no URLs."
            )
            title = f"COPING ACTIVITIES & NEXT STEPS (success probability: {success_probability:.0%})"

        return (
            f"You are an expert counseling assistant.\n"
            f"{base_header}\n"
            f"{title}:\n"
            f"INSTRUCTIONS: {instructions}\n\n"
            f"PARAGRAPH:"
        )

    def _postprocess_paragraph(self, text: str) -> str:
        # Collapse whitespace and strip numbering/bullets if any
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'(\d+\.|[-•*])\s+', '', text)
        # remove hashtags/handles and excessive symbols
        text = re.sub(r'[#@]+', '', text)
        text = re.sub(r'["“”]{2,}', '"', text)
        text = re.sub(r'[\*_/\\]{2,}', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        # Remove non-counseling/news artifacts
        lower = text.lower()
        if any(x in lower for x in ['published', 'all rights reserved', 'subscribe', 'breaking news', 'http://', 'https://', 'www.', 'share this article', '#image', 'book offers']):
            return ''
        return text

    def _is_valid_paragraph(self, text: str, user_data: Dict) -> bool:
        if not text or len(text) < 60:
            return False
        bad_tokens = ['#', 'subscribe', 'share this article', 'breaking news']
        if any(bt in text.lower() for bt in bad_tokens):
            return False
        # Require at least two user-specific anchors present to ensure grounding
        anchors = [
            user_data.get('current_mood', ''),
            user_data.get('challenges', ''),
            user_data.get('goals', ''),
            user_data.get('emotional_state', '')
        ]
        anchors = [a.strip().split(' ')[0].lower() for a in anchors if isinstance(a, str) and len(a.strip()) > 0]
        present = sum(1 for a in anchors if a and a in text.lower())
        return present >= 1  # at least one anchor present
    
    def _get_fallback_paragraph(self, section: str, user_data: Dict) -> str:
        """Provide fallback counseling paragraphs when AI generation fails"""
        current_mood = user_data.get('current_mood', '')
        stress_level = user_data.get('stress_level', 5)
        sleep_quality = user_data.get('sleep_quality', 5)
        social_support = user_data.get('social_support', 5)
        emotional_state = user_data.get('emotional_state', '')
        challenges = user_data.get('challenges', '')
        goals = user_data.get('goals', '')
        life_changes = user_data.get('life_changes', '')
        coping_mechanisms = user_data.get('coping_mechanisms', '')
        
        fallbacks = {
            'main': f"As a counselor, you should provide comprehensive, multi-faceted guidance to your client based on their current mood '{current_mood}' and stress level of {stress_level}/10. Given their specific challenges '{challenges}' and goals '{goals}', guide them toward building a sustainable self-care routine that adapts to their changing needs and circumstances. With their sleep quality at {sleep_quality}/10, help them prioritize establishing a consistent sleep schedule that supports their energy levels and emotional regulation - guide them in creating a relaxing bedtime routine that includes gentle stretching, reading, or meditation. Given their social support level of {social_support}/10, guide them in actively working on building meaningful connections through community activities, reaching out to friends, or joining groups that align with their interests. Their emotional state '{emotional_state}' suggests focusing your counseling on emotional regulation techniques such as deep breathing exercises, progressive muscle relaxation, or mindfulness meditation. For their life changes '{life_changes}', guide them in creating a structured approach to managing transitions by maintaining familiar routines while gradually incorporating new elements. Help them build upon their existing coping mechanisms '{coping_mechanisms}' by expanding and refining them, and guide them in adding new strategies that complement what already works for them. Guide them to start with morning breathing exercises for 5-10 minutes to set a positive, grounded tone for their day, and help them maintain consistency in their self-care practices. Encourage them to engage in activities that genuinely bring them joy and fulfillment, whether that's creative expression, physical movement, or meaningful social connections. Guide them in keeping a detailed mood and activity journal to track patterns, identify triggers, and discover what coping strategies work best for their specific situation. This self-awareness practice will help them recognize early warning signs and implement preventive measures before stress becomes overwhelming. Guide them to understand that healing and growth happen gradually through small, consistent steps, and help them be patient and compassionate with themselves throughout this process. Additionally, guide them in exploring different therapeutic techniques such as guided imagery, journaling, or gentle physical activities to find what resonates most with their personality and lifestyle. Help them build a comprehensive toolkit of coping strategies that ensures they have multiple options available when facing different types of challenges, and guide them in adapting their approach based on their current mood, energy level, and circumstances.",
            
            'emotions': f"As a counselor, guide your client in understanding that their emotional wellbeing is fundamental to their overall health and happiness, and help them develop strong emotional intelligence skills that will serve them throughout their entire life. Given their current mood '{current_mood}', stress level of {stress_level}/10, and emotional state '{emotional_state}', guide them in building comprehensive emotional awareness and regulation skills that help them navigate both everyday stressors and major life challenges. Teach them the 5-4-3-2-1 grounding technique when they feel overwhelmed: guide them to identify 5 things they can see, 4 they can touch, 3 they can hear, 2 they can smell, and 1 they can taste to bring themselves back to the present moment and interrupt anxious or negative thought patterns. With their sleep quality at {sleep_quality}/10, guide them in understanding how sleep affects their emotional regulation - help them recognize that poor sleep can significantly impact their ability to manage emotions effectively, so guide them in prioritizing sleep hygiene practices. Given their social support level of {social_support}/10, guide them in building emotional connections and learning to express their feelings in constructive ways through conversations with trusted friends or family members, or through creative outlets like journaling, art, or music. Help them develop healthy emotional boundaries that protect their energy and wellbeing while still allowing for meaningful connections with others. Guide them in building emotional resilience through regular mindfulness practices, self-compassion exercises, and activities that help them process and understand their emotions rather than suppressing them. Teach them emotional regulation techniques such as deep breathing, progressive muscle relaxation, or gentle physical movement when they notice intense emotions arising. Guide them to understand that all emotions are valid and temporary - help them see that the goal is not to eliminate difficult feelings but to develop healthy ways to navigate them, learn from them, and build lasting emotional wellness. Additionally, guide them in exploring emotional processing techniques like writing letters to express unspoken feelings, creating art to externalize emotions, or engaging in physical activities that help release emotional tension. Help them understand that building emotional intelligence is a lifelong journey that will enhance their relationships, decision-making, and overall life satisfaction, and guide them through this process, especially during life changes '{life_changes}' when emotions can be more intense and complex.",
            
            'insights': f"Based on your responses, you're demonstrating important self-awareness about your emotional state, which is a strong foundation for positive change and personal growth. Your willingness to reflect on your challenges '{challenges}' and goals '{goals}' shows emotional intelligence and readiness for growth, which are key predictors of successful outcomes in therapeutic work. With your stress level at {stress_level}/10 and sleep quality at {sleep_quality}/10, there's a clear opportunity to improve your overall wellbeing through targeted interventions. Many people with similar patterns find that consistent daily practices, such as mindfulness meditation, gentle exercise, or creative expression, significantly improve their overall wellbeing over time. The key is finding practices that feel authentic and sustainable for your unique personality and lifestyle, especially considering your current mood '{current_mood}' and emotional state '{emotional_state}'. Given your social support level of {social_support}/10, consider how building stronger connections might enhance your emotional resilience and provide additional resources for coping with challenges. Consider tracking your mood, activities, and coping strategies for a week to identify patterns and discover what works best for your specific situation. This self-monitoring approach often reveals valuable insights about triggers, effective coping mechanisms, and optimal timing for different wellness activities. You may discover that certain times of day, environmental factors, or social situations significantly impact your emotional state, allowing you to make proactive adjustments. Additionally, pay attention to the relationship between your physical health and emotional wellbeing, as they are deeply interconnected. Regular sleep, nutrition, and physical activity can have profound effects on your mood and stress levels. Remember that progress in emotional wellness is rarely linear - there will be ups and downs, and that's completely normal. The important thing is to maintain consistency with your practices and be kind to yourself during challenging periods. Consider setting small, achievable goals and celebrating your progress along the way, as this positive reinforcement can help maintain motivation and build confidence in your ability to manage your emotional wellbeing, especially during life changes '{life_changes}' when stability and routine become even more important.",
            
            'next_steps': f"As a counselor, guide your client based on their specific situation with stress level {stress_level}/10, sleep quality {sleep_quality}/10, social support {social_support}/10, current mood '{current_mood}', challenges '{challenges}', goals '{goals}', life changes '{life_changes}', and existing coping mechanisms '{coping_mechanisms}'. Guide them in implementing comprehensive, tailored coping activities immediately. For the next 24 hours, help them commit to one specific self-care action that directly addresses their current challenges - guide them toward a 10-15 minute walk in nature, calling a supportive friend or family member, or engaging in a creative activity that brings them joy and helps them process their emotions. This immediate action will help them feel more grounded and provide a sense of accomplishment. Given their sleep quality of {sleep_quality}/10, guide them this week in establishing one new sleep hygiene routine that supports better rest, such as going to bed 30 minutes earlier, creating a relaxing pre-sleep ritual, or limiting screen time before bed. With their social support level of {social_support}/10, guide them in making one meaningful social connection this week - help them reach out to a friend, join a community activity, or have a meaningful conversation with someone they trust. For their goals '{goals}', guide them in breaking them down into small, manageable steps and help them commit to one specific action this week that moves them closer to achieving them. Given their life changes '{life_changes}', guide them in creating a simple routine that provides stability and comfort during this transition period. Help them build upon their existing coping mechanisms '{coping_mechanisms}' by expanding them or adding complementary strategies. Guide them to understand that if they feel overwhelmed, it's perfectly okay to take things one small step at a time and to ask for support when they need it. Guide them in setting a weekly check-in with themselves to assess their progress and adjust their coping strategies as needed. Additionally, help them identify one person in their life who can serve as an accountability partner for their wellness goals, someone who will check in with them regularly and provide encouragement. Guide them in creating a simple tracking system for their mood and activities, even if it's just a few notes in their phone or a small notebook. This will help them identify patterns and celebrate their progress. Guide them to understand that building new habits takes time and patience, so help them be gentle with themselves if they miss a day or struggle with consistency. Help them see that the goal is progress, not perfection, and that every small step they take toward better emotional wellbeing is valuable and meaningful."
        }
        
        return fallbacks.get(section, "I'm here to support you on your wellness journey. Consider reaching out to a mental health professional if you need additional support.")
    
    def _create_counseling_prompt(self, user_data: Dict, ml_insights: Dict) -> str:
        """Create a comprehensive prompt for counseling recommendations"""
        
        # Extract user information
        current_mood = user_data.get('current_mood', '')
        stress_level = user_data.get('stress_level', 5)
        sleep_quality = user_data.get('sleep_quality', 5)
        social_support = user_data.get('social_support', 5)
        emotional_state = user_data.get('emotional_state', '')
        challenges = user_data.get('challenges', '')
        goals = user_data.get('goals', '')
        life_changes = user_data.get('life_changes', '')
        coping_mechanisms = user_data.get('coping_mechanisms', '')
        
        # Extract ML insights
        risk_level = ml_insights.get('risk_level', 'medium')
        text_emotions = ml_insights.get('text_emotions', {})
        text_sentiment = ml_insights.get('text_sentiment', {})
        user_cluster = ml_insights.get('user_cluster', 0)
        success_probability = ml_insights.get('success_probability', 0.7)
        
        # Create the prompt (constrained for counseling; forbid news/date/url noise)
        prompt = f"""You are an expert counseling assistant. Based on the user's context below, write a single cohesive paragraph of practical, real-life counseling advice tailored to the user. Ground every suggestion in their described situation and goals, and keep the tone supportive and specific.

USER PROFILE:
Current Mood: {current_mood}
Stress Level: {stress_level}/10
Sleep Quality: {sleep_quality}/10
Social Support: {social_support}/10
Emotional State: {emotional_state}
Challenges: {challenges}
Goals: {goals}
Life Changes: {life_changes}
Coping Mechanisms: {coping_mechanisms}

INSTRUCTIONS:
- Output exactly ONE paragraph (6–10 sentences), no lists or numbering.
- Refer directly to the user's inputs (mood, challenges, goals, routines) and suggest concrete steps.
- Include morning/evening routine ideas, a small accountability step, and one reflective question.
- Avoid dates, URLs, news/article references, and external sources.
- Do not mention being an AI or a language model.
- Supportive, professional tone; no diagnoses.

PARAGRAPH:"""
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using TinyLlama-Chat"""
        try:
            # Generate text with optimized parameters for speed
            result = self.generator(
                prompt,
                max_new_tokens=self.model_config['max_new_tokens'],
                min_new_tokens=self.model_config['min_new_tokens'],
                temperature=self.model_config['temperature'],
                top_p=self.model_config['top_p'],
                top_k=self.model_config['top_k'],
                do_sample=self.model_config['do_sample'],
                no_repeat_ngram_size=self.model_config['no_repeat_ngram_size'],
                repetition_penalty=self.model_config['repetition_penalty'],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1  # Only generate one sequence for speed
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            
            # Remove the original prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"⚠️ TinyLlama-Chat text generation error: {e}")
            return ""
    
    def _parse_recommendations(self, generated_text: str) -> List[str]:
        """Parse generated text into structured recommendations"""
        if not generated_text:
            return []
        
        # Expect a single paragraph; collapse to one item
        recommendations = []
        
        # Clean to a single paragraph (remove lists/numbering if any slipped in)
        par = generated_text
        par = re.sub(r'\n+', ' ', par)
        par = re.sub(r'(\d+\.|[-•*])\s+', '', par)
        par = re.sub(r'\s{2,}', ' ', par).strip()
        
        # Domain filtering: remove likely non-counseling/news noise and artifacts
        lower = par.lower()
        if any(bad in lower for bad in [
            'published', 'copyright', 'all rights reserved', 'subscribe', 'follow us', 'breaking news',
            'http://', 'https://', 'www.'
        ]):
            par = ''
        
        if par:
            recommendations.append(par)
        
        # Clean, filter, and limit to 4 domain-appropriate items
        recommendations = recommendations[:1]
        cleaned_recommendations = []
        
        for rec in recommendations:
            # Clean up the recommendation
            rec = re.sub(r'^\d+\.\s*', '', rec)  # Remove leading numbers
            rec = re.sub(r'^[-•*]\s*', '', rec)  # Remove bullet points
            rec = rec.strip()
            
            # Domain filtering: remove likely non-counseling/news noise
            lower = rec.lower()
            if any(bad in lower for bad in [
                'published', 'copyright', 'all rights reserved', 'subscribe', 'follow us', 'breaking news',
                'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
                'http://', 'https://', 'www.'
            ]):
                continue
            
            # Keep counseling-like items only
            if rec and len(rec) > 80 and len(rec) < 1200:
                cleaned_recommendations.append(rec)
        
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for rec in cleaned_recommendations:
            if rec not in seen:
                seen.add(rec)
                deduped.append(rec)
        
        # Ensure a single paragraph
        deduped = deduped[:1]
        return deduped
    
    def _get_fallback_recommendations(self, user_data: Dict, ml_insights: Dict) -> List[str]:
        """Fallback recommendations when TinyLlama is not available"""
        recommendations = []
        
        # Basic recommendations based on user data
        stress_level = user_data.get('stress_level', 5)
        sleep_quality = user_data.get('sleep_quality', 5)
        social_support = user_data.get('social_support', 5)
        challenges = user_data.get('challenges', '')
        
        if stress_level >= 7:
            recommendations.append("💡 Consider implementing daily stress management techniques such as deep breathing exercises or progressive muscle relaxation to help manage your high stress levels.")
        
        if sleep_quality <= 4:
            recommendations.append("💡 Focus on improving your sleep hygiene by establishing a consistent bedtime routine and creating a comfortable sleep environment.")
        
        if social_support <= 4:
            recommendations.append("💡 Building stronger social connections can significantly improve your well-being. Consider joining community groups or reaching out to friends and family.")
        
        if 'work' in challenges.lower() or 'job' in challenges.lower():
            recommendations.append("💡 Work-related stress is common. Consider time management techniques and setting clear boundaries between work and personal time.")
        
        if 'anxiety' in challenges.lower() or 'worried' in challenges.lower():
            recommendations.append("💡 For anxiety management, try grounding techniques like the 5-4-3-2-1 method or mindfulness meditation.")
        
        # Add ML-based insights
        risk_level = ml_insights.get('risk_level', 'medium')
        if risk_level == 'high':
            recommendations.append("💡 Given your current risk level, consider seeking professional counseling support for comprehensive care.")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def generate_emotion_specific_recommendations(self, user_data: Dict, emotions: Dict) -> List[str]:
        """Generate recommendations based on specific detected emotions"""
        if not self.generator:
            return self._get_emotion_fallback_recommendations(emotions)
        
        try:
            # Create emotion-specific prompt
            prompt = self._create_emotion_prompt(user_data, emotions)
            
            # Generate recommendations
            generated_text = self._generate_text(prompt)
            
            # Parse recommendations
            recommendations = self._parse_recommendations(generated_text)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ TinyLlama-Chat emotion-specific generation failed: {e}")
            return self._get_emotion_fallback_recommendations(emotions)
    
    def _create_emotion_prompt(self, user_data: Dict, emotions: Dict) -> str:
        """Create prompt for emotion-specific recommendations"""
        
        # Identify primary emotions
        primary_emotions = [emotion for emotion, count in emotions.items() if count > 0]
        
        prompt = f"""You are an expert AI counselor. The user has expressed the following emotions: {', '.join(primary_emotions)}.

USER CONTEXT:
Current Mood: {user_data.get('current_mood', '')}
Challenges: {user_data.get('challenges', '')}
Goals: {user_data.get('goals', '')}

Generate 3 specific, actionable recommendations to help the user manage these emotions effectively. Focus on evidence-based therapeutic techniques.

RECOMMENDATIONS:"""
        
        return prompt
    
    def _get_emotion_fallback_recommendations(self, emotions: Dict) -> List[str]:
        """Fallback emotion-specific recommendations"""
        recommendations = []
        
        if emotions.get('anxiety', 0) > 0:
            recommendations.append("💡 For anxiety management, try the 4-7-8 breathing technique: inhale for 4 counts, hold for 7, exhale for 8.")
        
        if emotions.get('depression', 0) > 0:
            recommendations.append("💡 For depression, consider daily physical activity and maintaining a regular sleep schedule.")
        
        if emotions.get('stress', 0) > 0:
            recommendations.append("💡 Progressive muscle relaxation can help reduce stress. Tense and relax each muscle group for 5 seconds.")
        
        if emotions.get('loneliness', 0) > 0:
            recommendations.append("💡 Combat loneliness by joining community activities or reaching out to one person you trust.")
        
        if emotions.get('anger', 0) > 0:
            recommendations.append("💡 For anger management, try the STOP technique: Stop, Take a breath, Observe your feelings, Proceed mindfully.")
        
        return recommendations[:3]
    
    def get_model_status(self) -> Dict:
        """Get status of TinyLlama-Chat model"""
        return {
            'model_available': self.generator is not None,
            'model_name': self.model_name,
            'device': self.device,
            'generation_mode': 'dynamic_text_generation',
            'fallback_mode': 'template_based' if not self.generator else 'ai_generated'
        }


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing TinyLlama-Chat Recommendation Engine...")
    
    # Initialize the engine
    engine = TinyLlamaRecommendationEngine()
    
    # Test with sample data
    test_user_data = {
        'current_mood': 'I feel very anxious and overwhelmed with work',
        'stress_level': 8,
        'sleep_quality': 3,
        'social_support': 4,
        'emotional_state': 'Stressed and worried about deadlines',
        'challenges': 'Too much work, not enough time, feeling isolated',
        'goals': 'Better work-life balance and stress management',
        'life_changes': 'New job responsibilities',
        'coping_mechanisms': 'Trying to exercise but struggling to find time'
    }
    
    test_ml_insights = {
        'risk_level': 'high',
        'text_emotions': {'anxiety': 2, 'stress': 1, 'loneliness': 1},
        'text_sentiment': {'sentiment': 'negative', 'confidence': 0.8},
        'user_cluster': 1,
        'success_probability': 0.75
    }
    
    print("\n🎯 Generating AI-powered recommendations...")
    recommendations = engine.generate_counseling_recommendations(test_user_data, test_ml_insights)
    
    print("\n📝 Generated Recommendations:")
    print("=" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📊 Model Status:")
    status = engine.get_model_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n✅ TinyLlama-Chat recommendation engine ready!")
