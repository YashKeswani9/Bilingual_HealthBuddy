# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import joblib
import random
import torch
import json
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from deep_translator import GoogleTranslator
from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionPredictDisease(Action):
    
    def name(self):
        return "action_predict_disease"
    
    def __init__(self):
        # Load the pre-trained model and vectorizer
        self.model = joblib.load('Utils/disease_predictor_model.pkl')
        self.vectorizer = joblib.load('Utils/vectorizer.pkl')

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        
        # Get entities from the latest message
        latest_message = tracker.latest_message
        entities = latest_message.get('entities', [])
        
        # Print debugging info
        print("\n=== DEBUG LANGUAGE INFO ===")
        print(f"Entities: {entities}")
        
        # Get language from slot
        language_slot = tracker.get_slot("language")
        print(f"Language slot: {language_slot}")
        
        # Find language entity
        language_entity = None
        for entity in entities:
            if entity.get('entity') == 'detected_language':
                language_entity = entity.get('value')
                break
        
        print(f"Language entity: {language_entity}")
        print("=========================\n")


        # Get the symptoms from the user input
        user_symptoms = tracker.latest_message.get('text')

        # Convert the symptoms text to the same format as the training data
        user_symptoms_tfidf = self.vectorizer.transform([user_symptoms])

        # Predict the disease using the trained Decision Tree model
        predicted_disease = self.model.predict(user_symptoms_tfidf)[0]

        # Default response in English
        response_text = f"""I understand that experiencing these symptoms can be concerning. 
Based on what you've shared, the predicted condition is: {predicted_disease}. 

Please keep in mind that my knowledge is limited, and this is just an AI-based prediction—not a medical diagnosis. 
For accurate medical advice, it's always best to consult a healthcare professional. 

Take care, and stay safe!"""

        # If detected language is French, translate the response
        if language_entity == "fr":
            response_text = GoogleTranslator(source='auto', target='fr').translate(response_text)

         # Send the predicted disease as a response
        dispatcher.utter_message(text=response_text)
        
        return []

class ActionWellnessTip(Action):
    def name(self):
        return "action_wellness_tip"
    
    def run(self, dispatcher, tracker, domain):
        wellness_tips = [
        "Stay Hydrated – Drink plenty of water throughout the day.",
        "Eat a Balanced Diet – Incorporate fruits, vegetables, whole grains, and lean proteins.",
        "Limit Processed Foods – Avoid excessive intake of sugary, salty, or fatty foods.",
        "Control Portion Sizes – Eating in moderation can prevent overeating.",
        "Exercise Regularly – Aim for at least 30 minutes of moderate exercise most days of the week.",
        "Stretch Daily – Stretching improves flexibility and reduces injury risk.",
        "Get Enough Sleep – Aim for 7-9 hours of sleep per night.",
        "Limit Caffeine – Consume caffeine in moderation, ideally earlier in the day.",
        "Quit Smoking – Stop smoking or avoid exposure to second-hand smoke.",
        "Limit Alcohol Consumption – Stick to the recommended limits (up to one drink for women, two for men per day).",
        "Stay Active Throughout the Day – Take breaks to walk around if you sit for long periods.",
        "Take the Stairs – Opt for stairs over elevators for a mini workout.",
        "Walk More – Take short walks after meals or during breaks.",
        "Balance Your Diet – Include protein, carbs, and healthy fats in each meal.",
        "Mind Your Posture – Good posture can reduce back pain and promote better digestion.",
        "Get Regular Checkups – Visit your healthcare provider regularly for screenings and preventive care.",
        "Maintain a Healthy Weight – Focus on healthy eating and regular activity to maintain a stable weight.",
        "Eat Fiber-Rich Foods – Help your digestion with fruits, vegetables, and whole grains.",
        "Add Healthy Fats – Include omega-3 rich foods like fish and nuts.",
        "Limit Red Meat – Reduce consumption of processed and fatty meats.",
        "Practice Deep Breathing – Helps reduce stress and improve mental clarity.",
        "Take Rest Days – Don’t overexert yourself; rest is essential for recovery.",
        "Use Sunscreen – Protect your skin from UV damage to avoid sunburn and skin cancer.",
        "Limit Screen Time – Too much screen time can cause eye strain and poor posture.",
        "Get a Massage – Massages help with relaxation and muscle tension.",
        "Practice Yoga or Pilates – Great for flexibility, core strength, and mental clarity.",
        "Maintain Healthy Skin – Cleanse and moisturize your skin regularly.",
        "Take a Walk Outdoors – Fresh air and nature are great for mental and physical health.",
        "Listen to Your Body – Rest when you feel fatigued and seek help when needed.",
        "Take Multivitamins – Fill in nutritional gaps with a multivitamin supplement, if needed.",
        "Practice Mindfulness – Stay present and reduce stress with mindfulness techniques.",
        "Set Realistic Goals – Break down big tasks into smaller, achievable ones.",
        "Stay Connected – Cultivate and maintain relationships with friends and family.",
        "Talk to Someone – Reach out for support if you’re feeling overwhelmed.",
        "Engage in Hobbies – Pursue activities that bring you joy and relaxation.",
        "Laugh Often – Laughter is a great stress reliever and mood booster.",
        "Limit Negative Thinking – Focus on the positives and practice gratitude.",
        "Avoid Negative People – Surround yourself with supportive and positive individuals.",
        "Practice Gratitude – Keep a gratitude journal or simply reflect on what you're thankful for.",
        "Seek Professional Help – A therapist or counselor can help you work through difficult emotions.",
        "Take Time for Yourself – Set aside 'me time' to recharge and focus on self-care.",
        "Challenge Yourself – Step out of your comfort zone to grow mentally.",
        "Meditate Daily – Meditation helps reduce anxiety, stress, and boosts concentration.",
        "Set Boundaries – Learn to say no and protect your mental well-being.",
        "Engage in Creative Activities – Painting, writing, or crafting can improve mental health.",
        "Get Adequate Sleep – A rested mind is better equipped to handle stress and challenges.",
        "Practice Positive Affirmations – Repeat kind and encouraging statements to boost confidence.",
        "Stay Organized – An organized environment promotes mental clarity.",
        "Take Mental Breaks – Step away from tasks to give your brain a rest.",
        "Stay Curious – Keep learning new things to engage your mind.",
        "Acknowledge Your Emotions – Don’t suppress feelings; express them constructively.",
        "Forgive Yourself – Let go of past mistakes and focus on growth.",
        "Practice Self-Compassion – Be kind and understanding toward yourself.",
        "Practice Empathy – Understand and share the feelings of others.",
        "Recognize Your Strengths – Reflect on what you’re good at and use it to boost your confidence.",
        "Let Go of Perfectionism – Embrace imperfection as part of life.",
        "Engage in Meaningful Conversations – Connect with others deeply rather than having superficial chats.",
        "Manage Your Stress – Use relaxation techniques like deep breathing or progressive muscle relaxation.",
        "Journal Regularly – Writing down thoughts can help process emotions and clarify your thoughts.",
        "Seek Professional Guidance – A counselor or life coach can provide valuable emotional support.",
        "Be Open to Change – Accept that change is a natural part of life and embrace new opportunities.",
        "Take Responsibility for Your Actions – Own your decisions and make adjustments as needed.",
        "Limit Exposure to Negative News – Too much negative media can impact your emotional health.",
        "Get Involved in Your Community – Volunteer or participate in local events for a sense of connection.",
        "Be Patient with Yourself – Give yourself time to heal emotionally and grow.",
        "Celebrate Small Wins – Recognize and celebrate achievements, big or small.",
        "Practice Non-judgment – Avoid judging others or yourself harshly.",
        "Cultivate a Growth Mindset – See challenges as opportunities for growth rather than obstacles.",
        "Develop Healthy Relationships – Surround yourself with people who support and uplift you.",
        "Express Appreciation – Regularly show gratitude to the people you care about.",
        "Communicate Openly – Effective communication helps to build stronger relationships.",
        "Be a Good Listener – Active listening shows you care and strengthens relationships.",
        "Make Time for Socializing – Balance work and personal time to nurture your social life.",
        "Resolve Conflicts Respectfully – Address issues calmly and constructively.",
        "Learn to Compromise – Healthy relationships often require give and take.",
        "Support Others – Be there for friends and family when they need help.",
        "Seek Diverse Perspectives – Embrace different viewpoints to broaden your understanding.",
        "Engage in Team Activities – Participate in group activities like sports or classes to build camaraderie.",
        "Show Kindness – Small acts of kindness can make a big difference in someone's day.",
        "Be Open to New Connections – Attend events or join groups to meet new people.",
        "Maintain a Clean Environment – A tidy living space promotes mental and physical well-being.",
        "Declutter Regularly – A clutter-free space can reduce stress and improve productivity.",
        "Get Fresh Air – Regularly ventilate your home and spend time outdoors.",
        "Create a Relaxing Space – Designate an area in your home for relaxation and unwinding.",
        "Reduce Noise Pollution – Minimize loud sounds to create a peaceful environment.",
        "Use Natural Cleaning Products – Avoid harmful chemicals and use eco-friendly alternatives.",
        "Be Conscious of Your Carbon Footprint – Limit energy use, recycle, and conserve water.",
        "Grow Plants – Indoor plants can purify the air and add a calming aesthetic.",
        "Use Eco-friendly Products – Choose sustainable options to protect the environment.",
        "Spend Time in Nature – Hike, visit parks, or simply enjoy nature for stress relief and physical health.",
        "Wear a Seatbelt – Always use your seatbelt while driving to prevent injuries.",
        "Avoid Risky Behaviors – Make safe choices to protect your health and well-being.",
        "Know Your Family Health History – Be aware of any hereditary conditions.",
        "Follow Safety Guidelines – Whether at work or home, always follow safety protocols.",
        "Get Vaccinated – Protect yourself and others by staying up to date on vaccinations.",
        "Wash Your Hands Regularly – Prevent the spread of germs by maintaining proper hygiene.",
        "Practice Safe Driving – Obey traffic laws and drive attentively.",
        "Limit Exposure to Toxic Substances – Avoid environmental toxins and pollutants when possible.",
        "Maintain Regular Health Screenings – Stay on top of important health checks like blood pressure, cholesterol, and glucose levels.",
        "Adopt Healthy Habits Early – Start adopting healthy habits early in life to ensure long-term health benefits."
    ]

        # Get entities from the latest message
        latest_message = tracker.latest_message
        entities = latest_message.get('entities', [])
        
        # Print debugging info
        print("\n=== DEBUG LANGUAGE INFO ===")
        print(f"Entities: {entities}")
        
        # Get language from slot
        language_slot = tracker.get_slot("language")
        print(f"Language slot: {language_slot}")
        
        # Find language entity
        language_entity = None
        for entity in entities:
            if entity.get('entity') == 'detected_language':
                language_entity = entity.get('value')
                break
        
        print(f"Language entity: {language_entity}")
        print("=========================\n")


        # Pick a random wellness tip
        tip = random.choice(wellness_tips)

         # If detected language is French, translate the response
        if language_entity == "fr":
            response_text = GoogleTranslator(source='auto', target='fr').translate(tip)

         # Send the predicted disease as a response
        dispatcher.utter_message(text=tip)
        
        return []


class ActionAskHealthQuestion(Action):
    def name(self):
        return "action_ask_health_question"

    def __init__(self):
        """Load MedQuAD dataset and initialize the embedding model.""" 
        medquad_path = "Utils/medquad_preprocessed.json"  
        with open(medquad_path, "r") as f:
            self.medquad_data = json.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight, efficient model
        self.questions = [entry["question"] for entry in self.medquad_data]
        self.answers = [entry["answer"] for entry in self.medquad_data]
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

        # Load FLAN-T5 Summarization Model
        self.summarizer = pipeline("summarization", model="google/flan-t5-base")


    def retrieve_health_answer(self, question):
        """Finds the most relevant answer using semantic similarity search."""
        user_embedding = self.model.encode(question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(user_embedding, self.embeddings)[0]
        best_match_idx = torch.argmax(similarities).item()

        # Ensure a confidence threshold before returning the answer
        if similarities[best_match_idx] < 0.8:  # Adjust threshold as needed
            return None  # No relevant answer found

        return self.answers[best_match_idx]

    def improve_response(self, answer):
        """Uses FLAN-T5 to summarize and refine the answer."""
        # try:
        #     response = self.summarizer(answer, max_length=100, min_length=30, do_sample=False)
        #     return response[0]["summary_text"]
        # except Exception as e:
        #     print(f"Summarization Error: {e}")
        #     return answer  # Return original answer if summarization fails
        try:
            word_count = len(answer.split())  # Count words in the original answer

            # Summarize only if answer exceeds 800 words
            if word_count > 800:
                response = self.summarizer(answer, max_length=800, min_length=500, do_sample=False)
                summarized_text = response[0]["summary_text"]
                
                # Ensure the summarized text has at least 500 words
                if len(summarized_text.split()) < 500:
                    return answer  # Return original if too short after summarizing
                
                return summarized_text

            return answer  # Return original answer if it's not too long

        except Exception as e:
            print(f"Summarization Error: {e}")
            return answer  # Return original answer if summarization fails

    def run(self, dispatcher, tracker, domain):
        user_question = tracker.latest_message.get("text")

        # Get entities from the latest message
        latest_message = tracker.latest_message
        entities = latest_message.get('entities', [])
        
        # Print debugging info
        print("\n=== DEBUG LANGUAGE INFO ===")
        print(f"Entities: {entities}")
        
        # Get language from slot
        language_slot = tracker.get_slot("language")
        print(f"Language slot: {language_slot}")
        
        # Find language entity
        language_entity = None
        for entity in entities:
            if entity.get('entity') == 'detected_language':
                language_entity = entity.get('value')
                break
        
        print(f"Language entity: {language_entity}")
        print("=========================\n")


        # Retrieve answer from MedQuAD
        retrieved_answer = self.retrieve_health_answer(user_question)

        if retrieved_answer:
            # Improve the response using summarization
            improved_answer = self.improve_response(retrieved_answer)
            
            # If detected language is French, translate the response
            if language_entity == "fr":
                improved_answer = GoogleTranslator(source='auto', target='fr').translate(improved_answer)
            
            dispatcher.utter_message(text=improved_answer)
        else:
            # Fallback response for unknown questions
            response_text = "I'm sorry, I don't have this information. Please consult a healthcare professional."
            
            # If detected language is French, translate the response
            if language_entity == "fr":
                response_text = GoogleTranslator(source='auto', target='fr').translate(response_text)

            dispatcher.utter_message(text=response_text)
            
        return []
