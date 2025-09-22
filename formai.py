from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
from dataclasses import dataclass, asdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import openai
import os
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Pydantic models for request/response (keeping existing ones)
class PersonProfileRequest(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    university: str = Field(..., description="University or institution")
    preferred_career: str = Field(..., description="Preferred career path")
    
    # Personality traits (1-5 scale)
    openness: int = Field(..., ge=1, le=5, description="Openness to experience")
    conscientiousness: int = Field(..., ge=1, le=5, description="Conscientiousness")
    extraversion: int = Field(..., ge=1, le=5, description="Extraversion")
    agreeableness: int = Field(..., ge=1, le=5, description="Agreeableness")
    neuroticism: int = Field(..., ge=1, le=5, description="Neuroticism")
    
    # Work values (1-6 ranking, 1=most important)
    income_importance: int = Field(..., ge=1, le=6, description="Importance of income (1=most important, 6=least)")
    impact_importance: int = Field(..., ge=1, le=6, description="Importance of making an impact")
    stability_importance: int = Field(..., ge=1, le=6, description="Importance of job stability")
    variety_importance: int = Field(..., ge=1, le=6, description="Importance of variety in work")
    recognition_importance: int = Field(..., ge=1, le=6, description="Importance of recognition")
    autonomy_importance: int = Field(..., ge=1, le=6, description="Importance of autonomy")
    
    # Skills (1-5 scale)
    math: int = Field(..., ge=1, le=5, description="Math skills")
    problem_solving: int = Field(..., ge=1, le=5, description="Problem solving skills")
    public_speaking: int = Field(..., ge=1, le=5, description="Public speaking skills")
    creative: int = Field(..., ge=1, le=5, description="Creative skills")
    working_with_people: int = Field(..., ge=1, le=5, description="People skills")
    writing: int = Field(..., ge=1, le=5, description="Writing & communication")
    tech_savvy: int = Field(..., ge=1, le=5, description="Technology skills")
    leadership: int = Field(..., ge=1, le=5, description="Leadership & management")
    networking: int = Field(..., ge=1, le=5, description="Networking & relationship building")
    programming: int = Field(..., ge=1, le=5, description="Programming proficiency")
    empathy: int = Field(..., ge=1, le=5, description="Empathy")
    time_management: int = Field(..., ge=1, le=5, description="Time management")
    attention_to_detail: int = Field(..., ge=1, le=5, description="Attention to detail")
    project_management: int = Field(..., ge=1, le=5, description="Project management")
    research: int = Field(..., ge=1, le=5, description="Research skills")
    teamwork: int = Field(..., ge=1, le=5, description="Teamwork")
    
    # Interests (multiple selection)
    interests: List[str] = Field(..., description="List of interests from: investigative, social, artistic, enterprising, realistic, conventional")

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    result: Optional[Dict] = None
    analysis_date: Optional[str] = None

@dataclass
class PersonProfile:
    name: str
    email: str
    university: str
    personality: Dict[str, float]
    work_values: Dict[str, float]
    skills: Dict[str, float]
    interests: List[str]
    preferred_career: str

class AICareerMatcher:
    def __init__(self):
        # Enhanced O*NET Job Database with similar roles mapping
        self.onet_jobs = {
            "Software Developer": {
                "onet_code": "15-1252.00",
                "skills": {
                    "math": 3.9, "problem_solving": 4.4, "tech_savvy": 4.6, 
                    "programming": 4.5, "attention_to_detail": 4.1, "research": 3.8
                },
                "work_values": {
                    "income": 4.1, "impact": 3.8, "stability": 4.0, "variety": 4.2, 
                    "recognition": 3.6, "autonomy": 4.3
                },
                "interests": {
                    "investigative": 4.4, "realistic": 3.2, "artistic": 2.8, 
                    "social": 2.1, "enterprising": 2.9, "conventional": 3.3
                },
                "work_styles": {
                    "analytical_thinking": 4.5, "attention_to_detail": 4.3, "dependability": 4.1,
                    "persistence": 4.0, "stress_tolerance": 3.8, "adaptability": 4.2
                },
                "required_skills": ["Programming", "Problem Solving", "Math", "Tech-Savvy", "Attention to Detail"],
                "similar_roles": ["Data Scientist", "DevOps Engineer", "Full Stack Developer", "Mobile App Developer", "Systems Architect"],
                "job_keywords": ["coding", "development", "technology", "innovation"]
            },
            "Marketing Manager": {
                "onet_code": "11-2021.00",
                "skills": {
                    "public_speaking": 4.3, "creative": 4.0, "networking": 4.2, 
                    "leadership": 4.1, "writing": 3.9, "working_with_people": 4.0
                },
                "work_values": {
                    "recognition": 4.3, "impact": 4.1, "variety": 4.2, "autonomy": 4.0, 
                    "income": 4.0, "stability": 3.5
                },
                "interests": {
                    "enterprising": 4.7, "artistic": 3.4, "social": 3.8, 
                    "investigative": 3.2, "conventional": 3.1, "realistic": 1.9
                },
                "work_styles": {
                    "leadership": 4.4, "stress_tolerance": 4.0, "adaptability": 4.3,
                    "initiative": 4.2, "achievement": 4.1, "social_orientation": 4.0
                },
                "required_skills": ["Public Speaking", "Creative", "Networking", "Leadership"],
                "similar_roles": ["Brand Manager", "Digital Marketing Specialist", "Product Marketing Manager", "Content Marketing Manager", "Growth Hacker"],
                "job_keywords": ["branding", "campaigns", "strategy", "growth"]
            },
            "Business Engineer": {
                "onet_code": "17-2199.00",
                "skills": {
                    "problem_solving": 4.3, "project_management": 4.1, "math": 4.0, 
                    "research": 4.2, "tech_savvy": 3.8, "writing": 3.7
                },
                "work_values": {
                    "impact": 4.2, "variety": 4.0, "autonomy": 3.8, "recognition": 3.7, 
                    "income": 4.0, "stability": 3.9
                },
                "interests": {
                    "investigative": 4.3, "enterprising": 4.0, "realistic": 3.5, 
                    "conventional": 3.2, "artistic": 2.5, "social": 2.8
                },
                "work_styles": {
                    "analytical_thinking": 4.4, "dependability": 4.2, "persistence": 4.1,
                    "achievement": 4.0, "adaptability": 3.9, "initiative": 4.0
                },
                "required_skills": ["Problem Solving", "Project Management", "Math", "Research", "Tech-Savvy"],
                "similar_roles": ["Business Analyst", "Management Consultant", "Operations Manager", "Strategy Consultant", "Process Improvement Specialist"],
                "job_keywords": ["optimization", "analysis", "efficiency", "solutions"]
            },
            "Elementary School Teacher": {
                "onet_code": "25-2021.00",
                "skills": {
                    "working_with_people": 4.5, "empathy": 4.4, "public_speaking": 4.2, 
                    "creative": 4.0, "time_management": 4.1, "writing": 3.8
                },
                "work_values": {
                    "impact": 4.5, "stability": 4.2, "autonomy": 3.2, "recognition": 3.7, 
                    "income": 3.0, "variety": 3.8
                },
                "interests": {
                    "social": 4.8, "artistic": 3.6, "investigative": 3.4, 
                    "conventional": 3.0, "enterprising": 2.8, "realistic": 2.1
                },
                "work_styles": {
                    "social_orientation": 4.6, "dependability": 4.4, "stress_tolerance": 4.0,
                    "adaptability": 4.2, "persistence": 4.1, "concern_for_others": 4.5
                },
                "required_skills": ["Working with People", "Empathy", "Public Speaking", "Creative", "Time Management"],
                "similar_roles": ["Middle School Teacher", "Curriculum Developer", "Education Coordinator", "Learning Specialist", "Academic Coach"],
                "job_keywords": ["education", "nurturing", "development", "learning"]
            },
            "Registered Nurse": {
                "onet_code": "29-1141.00",
                "skills": {
                    "empathy": 4.5, "working_with_people": 4.4, "attention_to_detail": 4.3, 
                    "problem_solving": 4.1, "time_management": 4.2, "teamwork": 4.3
                },
                "work_values": {
                    "impact": 4.4, "stability": 4.1, "recognition": 3.8, "income": 3.8, 
                    "autonomy": 3.4, "variety": 3.6
                },
                "interests": {
                    "social": 4.6, "investigative": 4.0, "realistic": 2.9, 
                    "conventional": 3.2, "artistic": 2.3, "enterprising": 2.8
                },
                "work_styles": {
                    "concern_for_others": 4.7, "stress_tolerance": 4.3, "dependability": 4.4,
                    "attention_to_detail": 4.3, "adaptability": 4.1, "cooperation": 4.2
                },
                "required_skills": ["Empathy", "Working with People", "Attention to Detail", "Problem Solving", "Time Management"],
                "similar_roles": ["Nurse Practitioner", "Clinical Nurse Specialist", "Charge Nurse", "Case Manager", "Healthcare Coordinator"],
                "job_keywords": ["healthcare", "compassion", "care", "wellness"]
            }
        }

    def create_profile_from_request(self, request: PersonProfileRequest) -> PersonProfile:
        """Create PersonProfile from API request"""
        personality = {
            "openness": request.openness,
            "conscientiousness": request.conscientiousness,
            "extraversion": request.extraversion,
            "agreeableness": request.agreeableness,
            "neuroticism": request.neuroticism
        }
        
        work_values = {
            "income": 7 - request.income_importance,
            "impact": 7 - request.impact_importance,
            "stability": 7 - request.stability_importance,
            "variety": 7 - request.variety_importance,
            "recognition": 7 - request.recognition_importance,
            "autonomy": 7 - request.autonomy_importance
        }
        
        skills = {
            "math": request.math,
            "problem_solving": request.problem_solving,
            "public_speaking": request.public_speaking,
            "creative": request.creative,
            "working_with_people": request.working_with_people,
            "writing": request.writing,
            "tech_savvy": request.tech_savvy,
            "leadership": request.leadership,
            "networking": request.networking,
            "programming": request.programming,
            "empathy": request.empathy,
            "time_management": request.time_management,
            "attention_to_detail": request.attention_to_detail,
            "project_management": request.project_management,
            "research": request.research,
            "teamwork": request.teamwork
        }
        
        return PersonProfile(
            name=request.name,
            email=request.email,
            university=request.university,
            personality=personality,
            work_values=work_values,
            skills=skills,
            interests=request.interests,
            preferred_career=request.preferred_career
        )

    async def generate_ai_insights(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate AI-powered insights for a specific job match"""
        job = self.onet_jobs[job_name]
        
        # Prepare user data summary for AI
        user_summary = self._prepare_user_summary(profile, match_data)
        
        # Generate AI summary
        ai_summary = await self._generate_ai_summary(profile, job_name, job, match_data)
        
        # Generate top keywords and O-NET categories
        keywords = await self._generate_keywords(job_name, job)
        onet_categories = await self._generate_onet_categories(job_name, job)
        
        # Generate action plan
        action_plan = await self._generate_action_plan(profile, job_name, match_data)
        
        # Generate career story
        career_story = await self._generate_career_story(profile, job_name, match_data)
        
        # Generate interview insights
        interview_insights = await self._generate_interview_insights(profile, job_name, match_data)
        
        return {
            "ai_summary": ai_summary,
            "keywords": keywords,
            "onet_categories": onet_categories,
            "action_plan": action_plan,
            "career_story": career_story,
            "interview_insights": interview_insights,
            "similar_roles": job.get("similar_roles", [])
        }

    def _prepare_user_summary(self, profile: PersonProfile, match_data: Dict) -> str:
        """Prepare a concise summary of user data for AI processing"""
        top_skills = sorted(profile.skills.items(), key=lambda x: x[1], reverse=True)[:5]
        top_values = sorted(profile.work_values.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return f"""
        User: {profile.name}
        Top Skills: {', '.join([f'{skill}: {level}/5' for skill, level in top_skills])}
        Top Work Values: {', '.join([f'{value}: {level}/6' for value, level in top_values])}
        Interests: {', '.join(profile.interests)}
        Overall Match: {match_data['overall_match']}%
        Strengths: {len(match_data['strengths'])} identified
        Improvement Areas: {len(match_data['improvements'])} identified
        """

    async def _generate_ai_summary(self, profile: PersonProfile, job_name: str, job: Dict, match_data: Dict) -> str:
        """Generate AI summary of job fit"""
        prompt = f"""
        Create a personalized 2-3 paragraph summary for {profile.name} regarding their fit for the {job_name} position.
        
        Key Information:
        - Overall Match: {match_data['overall_match']}%
        - Skills Match: {match_data['breakdown']['skills_match']}%
        - Values Match: {match_data['breakdown']['values_match']}%
        - Interests Match: {match_data['breakdown']['interests_match']}%
        - Work Styles Match: {match_data['breakdown']['work_styles_match']}%
        
        User Strengths: {', '.join(match_data['strengths'])}
        Areas for Improvement: {len(match_data['improvements'])} key areas identified
        
        Write in a encouraging, professional tone that:
        1. Acknowledges their strengths and natural fit
        2. Addresses any gaps constructively
        3. Provides realistic outlook on their candidacy
        4. Motivates action toward career goals
        
        Keep it concise but insightful.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"AI analysis temporarily unavailable. Based on your {match_data['overall_match']}% match score, you show strong potential for this role with {len(match_data['strengths'])} key strengths identified."

    async def _generate_keywords(self, job_name: str, job: Dict) -> List[str]:
        """Generate 4 relevant keywords for the job"""
        if "job_keywords" in job:
            return job["job_keywords"][:4]
        
        prompt = f"""
        Generate exactly 4 keywords that best represent the {job_name} role.
        These should be:
        - Single words or short phrases (2-3 words max)
        - Industry-relevant
        - What someone would associate with this career
        - Professional and specific
        
        Return only the keywords, separated by commas.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5
            )
            keywords = [k.strip() for k in response.choices[0].message.content.strip().split(',')]
            return keywords[:4]
        except Exception:
            return ["professional", "skilled", "dedicated", "growth-oriented"]

    async def _generate_onet_categories(self, job_name: str, job: Dict) -> Dict[str, List[str]]:
        """Generate 3 words for each O-NET category"""
        categories = {
            "skills": list(job["skills"].keys())[:3],
            "work_values": list(job["work_values"].keys())[:3],
            "work_styles": list(job.get("work_styles", {}).keys())[:3],
            "interests": list(job["interests"].keys())[:3]
        }
        
        # Clean up the categories
        for category, items in categories.items():
            categories[category] = [item.replace('_', ' ').title() for item in items]
        
        return categories

    async def _generate_action_plan(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate personalized action plan"""
        improvements = match_data.get('improvements', [])
        
        prompt = f"""
        Create a focused action plan for {profile.name} to become a competitive candidate for {job_name}.
        
        Current gaps to address:
        {chr(10).join([f"- {imp['skill']}: Gap of {imp['required_level'] - imp['current_level']:.1f} points" for imp in improvements[:3]])}
        
        Provide:
        1. Top Needs (3 priority areas)
        2. Action Items (4-5 specific, actionable steps)
        
        Make it practical and achievable within 3-6 months.
        Format as JSON with "top_needs" and "action_items" arrays.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            # Try to extract JSON, fallback if needed
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # Fallback structure
            return {
                "top_needs": [
                    f"Develop {improvements[0]['skill']} skills" if improvements else "Enhance core competencies",
                    "Build relevant experience",
                    "Network in target industry"
                ],
                "action_items": [
                    "Take online courses in key skill areas",
                    "Seek relevant projects or volunteer opportunities", 
                    "Connect with professionals in the field",
                    "Practice interviewing for this role type",
                    "Build a portfolio showcasing relevant work"
                ]
            }
            
        except Exception:
            return {
                "top_needs": ["Skill development", "Experience building", "Industry networking"],
                "action_items": ["Complete relevant courses", "Gain hands-on experience", "Build professional network", "Prepare interview materials"]
            }

    async def _generate_career_story(self, profile: PersonProfile, job_name: str, match_data: Dict) -> str:
        """Generate a compelling career narrative"""
        strengths = match_data.get('strengths', [])
        
        prompt = f"""
        Write a compelling 2-paragraph career story for {profile.name} pursuing {job_name}.
        
        Their strengths: {', '.join(strengths[:4])}
        Match score: {match_data['overall_match']}%
        
        Create a narrative that:
        1. First paragraph: Connects their background and strengths to this career path naturally
        2. Second paragraph: Shows vision for their future growth and impact in this role
        
        Write in first person as if they're telling their story.
        Be authentic, confident, and forward-looking.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return f"My journey toward {job_name} has been shaped by my natural strengths and genuine passion for this field. Through my experiences, I've developed key capabilities that align well with what this role demands. I'm excited about the opportunity to bring my skills and perspective to make a meaningful impact in this position and grow alongside the organization."

    async def _generate_interview_insights(self, profile: PersonProfile, job_name: str, match_data: Dict) -> Dict:
        """Generate interview-ready insights"""
        strengths = match_data.get('strengths', [])
        
        prompt = f"""
        Create interview preparation insights for {profile.name} applying for {job_name}.
        
        Their key strengths: {', '.join(strengths[:4])}
        
        Provide:
        1. Key Selling Points (3 main strengths to emphasize)
        2. Story Examples (3 specific scenarios they should prepare)
        3. Questions to Ask (3 thoughtful questions for the interviewer)
        
        Format as JSON with these three arrays.
        Make it specific and actionable for interview success.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.6
            )
            
            content = response.choices[0].message.content.strip()
            try:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # Fallback
            return {
                "key_selling_points": [
                    "Strong analytical and problem-solving abilities",
                    "Excellent communication and interpersonal skills", 
                    "Proven ability to adapt and learn quickly"
                ],
                "story_examples": [
                    "Time you solved a complex problem using analytical thinking",
                    "Situation where you collaborated effectively with a team",
                    "Example of learning a new skill quickly and applying it successfully"
                ],
                "questions_to_ask": [
                    "What does success look like in this role after the first year?",
                    "What are the biggest challenges facing the team right now?",
                    "How does this role contribute to the organization's strategic goals?"
                ]
            }
            
        except Exception:
            return {
                "key_selling_points": ["Strong foundational skills", "Growth mindset", "Team collaboration"],
                "story_examples": ["Problem-solving example", "Teamwork scenario", "Learning achievement"],
                "questions_to_ask": ["Role expectations", "Team challenges", "Growth opportunities"]
            }

    def calculate_job_match(self, profile: PersonProfile, job_name: str) -> Dict:
        """Calculate comprehensive match percentage and details for a specific job"""
        job = self.onet_jobs[job_name]
        
        # Skills match (30% weight)
        skills_score = self._calculate_skills_match(profile.skills, job["skills"])
        
        # Work values match (25% weight) 
        values_score = self._calculate_values_match(profile.work_values, job["work_values"])
        
        # Interests match (20% weight)
        interests_score = self._calculate_interests_match(profile.interests, job["interests"])
        
        # Work styles match (25% weight)
        work_styles_score = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
        
        # Overall match
        overall_match = (skills_score * 0.3 + values_score * 0.25 + 
                        interests_score * 0.2 + work_styles_score * 0.25) * 100
        
        # Identify strengths and improvement areas
        strengths, improvements = self._identify_strengths_improvements(profile, job)
        
        return {
            "job_name": job_name,
            "onet_code": job["onet_code"],
            "overall_match": round(overall_match, 1),
            "breakdown": {
                "skills_match": round(skills_score * 100, 1),
                "values_match": round(values_score * 100, 1),
                "interests_match": round(interests_score * 100, 1),
                "work_styles_match": round(work_styles_score * 100, 1)
            },
            "strengths": strengths,
            "improvements": improvements,
            "required_skills": job["required_skills"]
        }

    def _calculate_skills_match(self, user_skills: Dict, job_skills: Dict) -> float:
        """Calculate skills compatibility score"""
        scores = []
        
        for skill, importance in job_skills.items():
            user_level = user_skills.get(skill, 2.5)
            
            # Normalize both to 0-1 scale
            normalized_importance = importance / 5.0
            normalized_user = user_level / 5.0
            
            # Calculate match
            if normalized_user >= normalized_importance:
                score = 1.0
            else:
                gap = normalized_importance - normalized_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * normalized_importance)
        
        return sum(scores) / sum(job_skills.values()) * 5 if scores else 0

    def _calculate_values_match(self, user_values: Dict, job_values: Dict) -> float:
        """Calculate work values compatibility"""
        scores = []
        
        for value, job_importance in job_values.items():
            user_importance = user_values.get(value, 3.5)
            
            # Normalize to 0-1
            norm_job = job_importance / 5.0
            norm_user = user_importance / 6.0
            
            # Calculate similarity
            difference = abs(norm_job - norm_user)
            similarity = 1 - difference
            
            scores.append(similarity * norm_job)
        
        return sum(scores) / sum(job_values.values()) * 5 if scores else 0

    def _calculate_interests_match(self, user_interests: List[str], job_interests: Dict) -> float:
        """Calculate interests compatibility"""
        if not user_interests:
            return 0.5
        
        total_score = 0
        max_possible = 0
        
        for interest, importance in job_interests.items():
            max_possible += importance
            if interest in user_interests:
                total_score += importance
        
        return (total_score / max_possible) if max_possible > 0 else 0

    def _calculate_work_styles_match(self, personality: Dict, job_work_styles: Dict) -> float:
        """Calculate work styles compatibility based on personality traits"""
        if not job_work_styles:
            return 0.5
        
        # Map personality traits to work styles
        personality_to_work_style = {
            "analytical_thinking": personality.get("openness", 3),
            "attention_to_detail": personality.get("conscientiousness", 3),
            "dependability": personality.get("conscientiousness", 3),
            "leadership": personality.get("extraversion", 3),
            "stress_tolerance": 5 - personality.get("neuroticism", 3),
            "adaptability": personality.get("openness", 3),
            "social_orientation": personality.get("extraversion", 3),
            "achievement": personality.get("conscientiousness", 3),
            "initiative": personality.get("extraversion", 3),
            "persistence": personality.get("conscientiousness", 3),
            "concern_for_others": personality.get("agreeableness", 3),
            "cooperation": personality.get("agreeableness", 3)
        }
        
        scores = []
        for style, importance in job_work_styles.items():
            user_level = personality_to_work_style.get(style, 3.0)
            
            norm_importance = importance / 5.0
            norm_user = user_level / 5.0
            
            if norm_user >= norm_importance * 0.8:
                score = 1.0
            else:
                gap = (norm_importance * 0.8) - norm_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * norm_importance)
        
        return sum(scores) / sum(job_work_styles.values()) * 5 if scores else 0

    def _identify_strengths_improvements(self, profile: PersonProfile, job: Dict) -> Tuple[List[str], List[Dict]]:
        """Identify user strengths and areas for improvement"""
        strengths = []
        improvements = []
        
        skill_mapping = {
            "Programming": "programming",
            "Problem Solving": "problem_solving",
            "Public Speaking": "public_speaking",
            "Creative": "creative",
            "Working with People": "working_with_people",
            "Leadership": "leadership",
            "Networking": "networking",
            "Math": "math",
            "Tech-Savvy": "tech_savvy",
            "Empathy": "empathy",
            "Time Management": "time_management",
            "Attention to Detail": "attention_to_detail",
            "Project Management": "project_management",
            "Research": "research"
        }
        
        for skill_name in job["required_skills"]:
            skill_key = skill_mapping.get(skill_name, skill_name.lower().replace(' ', '_'))
            user_level = profile.skills.get(skill_key, 2.5)
            job_requirement = job["skills"].get(skill_key, 3.5)
            
            if user_level >= job_requirement:
                strengths.append(f"Strong {skill_name} skills (Level {user_level}/5)")
            elif user_level < job_requirement - 0.5:
                improvements.append({
                    "skill": skill_name,
                    "current_level": user_level,
                    "required_level": job_requirement,
                    "gap_severity": "High" if job_requirement - user_level > 1.5 else "Medium"
                })
        
        return strengths, improvements

    async def analyze_person_with_ai(self, profile: PersonProfile) -> Dict:
        """Analyze a person against all jobs and return enhanced matches with AI insights"""
        matches = []
        
        for job_name in self.onet_jobs.keys():
            # Calculate basic match
            match_result = self.calculate_job_match(profile, job_name)
            
            # Generate AI insights
            ai_insights = await self.generate_ai_insights(profile, job_name, match_result)
            
            # Combine results
            enhanced_match = {**match_result, **ai_insights}
            matches.append(enhanced_match)
        
        matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        return {
            "profile": asdict(profile),
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def generate_pdf_report(self, analysis_data: Dict, job_name: str) -> io.BytesIO:
        """Generate a comprehensive PDF report for a specific job match"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E3440')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#5E81AC')
        )
        
        # Find the specific job match
        job_match = None
        for match in analysis_data['matches']:
            if match['job_name'] == job_name:
                job_match = match
                break
        
        if not job_match:
            job_match = analysis_data['matches'][0]  # Fallback to top match
        
        # Build PDF content
        story = []
        
        # Title Page
        story.append(Paragraph(f"Career Analysis Report", title_style))
        story.append(Paragraph(f"{analysis_data['profile']['name']}", styles['Heading2']))
        story.append(Paragraph(f"Position: {job_match['job_name']}", styles['Heading3']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        story.append(Paragraph(job_match.get('ai_summary', 'AI summary not available'), styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Match Score Overview
        story.append(Paragraph("Match Analysis", heading_style))
        match_data = [
            ['Category', 'Score'],
            ['Overall Match', f"{job_match['overall_match']}%"],
            ['Skills Match', f"{job_match['breakdown']['skills_match']}%"],
            ['Values Match', f"{job_match['breakdown']['values_match']}%"],
            ['Interests Match', f"{job_match['breakdown']['interests_match']}%"],
            ['Work Styles Match', f"{job_match['breakdown']['work_styles_match']}%"]
        ]
        
        match_table = Table(match_data, colWidths=[3*inch, 2*inch])
        match_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D8DEE9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(match_table)
        story.append(Spacer(1, 20))
        
        # Keywords
        story.append(Paragraph("Key Focus Areas", heading_style))
        keywords_text = " • ".join(job_match.get('keywords', ['Professional', 'Skilled', 'Growth-oriented']))
        story.append(Paragraph(keywords_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # O-NET Categories
        story.append(Paragraph("O-NET Professional Profile", heading_style))
        onet_cats = job_match.get('onet_categories', {})
        for category, items in onet_cats.items():
            story.append(Paragraph(f"<b>{category.replace('_', ' ').title()}:</b> {', '.join(items)}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Action Plan
        action_plan = job_match.get('action_plan', {})
        story.append(Paragraph("Development Action Plan", heading_style))
        
        if 'top_needs' in action_plan:
            story.append(Paragraph("<b>Priority Development Areas:</b>", styles['Normal']))
            for i, need in enumerate(action_plan['top_needs'], 1):
                story.append(Paragraph(f"{i}. {need}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        if 'action_items' in action_plan:
            story.append(Paragraph("<b>Recommended Actions:</b>", styles['Normal']))
            for i, action in enumerate(action_plan['action_items'], 1):
                story.append(Paragraph(f"{i}. {action}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Page break
        story.append(PageBreak())
        
        # Career Story
        story.append(Paragraph("Your Career Narrative", heading_style))
        career_story = job_match.get('career_story', 'Career story not available')
        story.append(Paragraph(career_story, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Interview Insights
        story.append(Paragraph("Interview Preparation Guide", heading_style))
        interview_insights = job_match.get('interview_insights', {})
        
        if 'key_selling_points' in interview_insights:
            story.append(Paragraph("<b>Your Key Selling Points:</b>", styles['Normal']))
            for point in interview_insights['key_selling_points']:
                story.append(Paragraph(f"• {point}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        if 'story_examples' in interview_insights:
            story.append(Paragraph("<b>Stories to Prepare:</b>", styles['Normal']))
            for story_example in interview_insights['story_examples']:
                story.append(Paragraph(f"• {story_example}", styles['Normal']))
            story.append(Spacer(1, 10))
        
        if 'questions_to_ask' in interview_insights:
            story.append(Paragraph("<b>Strategic Questions to Ask:</b>", styles['Normal']))
            for question in interview_insights['questions_to_ask']:
                story.append(Paragraph(f"• {question}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Similar Roles
        story.append(Paragraph("Related Career Opportunities", heading_style))
        similar_roles = job_match.get('similar_roles', [])
        if similar_roles:
            roles_text = ", ".join(similar_roles)
            story.append(Paragraph(f"Consider exploring: {roles_text}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Report generated on {analysis_data['analysis_date']}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer


# Initialize FastAPI app
app = FastAPI(title="AI-Enhanced Career Matching API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI matcher
ai_matcher = AICareerMatcher()

@app.post("/analyze-profile-ai", response_model=AnalysisResponse)
async def analyze_profile_with_ai(request: PersonProfileRequest):
    """
    Analyze a person's profile with AI-enhanced insights and return comprehensive career matching results
    """
    try:
        # Create profile from request
        profile = ai_matcher.create_profile_from_request(request)
        
        # Analyze the profile with AI insights
        result = await ai_matcher.analyze_person_with_ai(profile)
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed profile for {profile.name} with AI insights",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@app.get("/download-report/{job_name}")
async def download_career_report(job_name: str, analysis_data: str):
    """
    Download a PDF report for a specific job match
    Note: In production, you'd want to store analysis results and retrieve by ID
    """
    try:
        # Parse the analysis data (in production, retrieve from database by ID)
        import json
        import urllib.parse
        
        decoded_data = urllib.parse.unquote(analysis_data)
        analysis_dict = json.loads(decoded_data)
        
        # Generate PDF
        pdf_buffer = ai_matcher.generate_pdf_report(analysis_dict, job_name)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=career_report_{job_name.replace(' ', '_')}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@app.post("/generate-job-insights")
async def generate_specific_job_insights(
    job_name: str,
    request: PersonProfileRequest
):
    """
    Generate AI insights for a specific job without full analysis
    """
    try:
        profile = ai_matcher.create_profile_from_request(request)
        match_data = ai_matcher.calculate_job_match(profile, job_name)
        ai_insights = await ai_matcher.generate_ai_insights(profile, job_name, match_data)
        
        return {
            "success": True,
            "job_name": job_name,
            "match_data": match_data,
            "ai_insights": ai_insights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job insights: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI-Enhanced Career Matching API",
        "version": "2.0.0",
        "features": [
            "AI-powered job fit summaries",
            "Personalized action plans",
            "Interview preparation insights",
            "Career story generation",
            "PDF report downloads",
            "Similar role recommendations"
        ],
        "endpoints": {
            "/analyze-profile-ai": "POST - Complete AI-enhanced profile analysis",
            "/generate-job-insights": "POST - Generate AI insights for specific job",
            "/download-report/{job_name}": "GET - Download PDF report",
            "/jobs": "GET - List available job types",
            "/health": "GET - Health check"
        }
    }

@app.get("/jobs")
async def get_jobs():
    """Get list of available job types with enhanced details"""
    jobs_info = {}
    for job_name, job_data in ai_matcher.onet_jobs.items():
        jobs_info[job_name] = {
            "onet_code": job_data["onet_code"],
            "required_skills": job_data["required_skills"],
            "similar_roles": job_data.get("similar_roles", []),
            "keywords": job_data.get("job_keywords", []),
            "top_work_values": sorted(job_data["work_values"].items(), key=lambda x: x[1], reverse=True)[:3],
            "top_interests": sorted(job_data["interests"].items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    return {
        "available_jobs": list(ai_matcher.onet_jobs.keys()),
        "job_details": jobs_info,
        "ai_features": [
            "Personalized fit analysis",
            "Career development roadmap", 
            "Interview preparation guide",
            "Professional narrative development"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with AI service status"""
    ai_status = "available"
    try:
        # Test OpenAI connection with a minimal request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        ai_status = "connected"
    except Exception:
        ai_status = "unavailable"
    
    return {
        "status": "healthy",
        "ai_service": ai_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    # Make sure to set OPENAI_API_KEY environment variable
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set!")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)