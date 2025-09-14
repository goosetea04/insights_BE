from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass, asdict

# Pydantic models for request/response
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

class CareerMatcher:
    def __init__(self):
        # O*NET Job Database with detailed metrics
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
                "improvement_tips": {
                    "Programming": "Take coding bootcamps or online courses in Python, Java, or JavaScript",
                    "Problem Solving": "Practice algorithmic thinking through LeetCode or HackerRank",
                    "Math": "Strengthen discrete mathematics, statistics, and linear algebra",
                    "Tech-Savvy": "Learn frameworks like React, Django, or cloud platforms (AWS, Azure)",
                    "Attention to Detail": "Practice code reviews and test-driven development"
                }
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
                "improvement_tips": {
                    "Public Speaking": "Join Toastmasters or take presentation skills workshops",
                    "Creative": "Develop creative thinking through design courses and brainstorming techniques",
                    "Networking": "Attend industry events and practice relationship-building strategies",
                    "Leadership": "Seek leadership opportunities and consider management training"
                }
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
                "improvement_tips": {
                    "Problem Solving": "Practice case studies and analytical frameworks used in consulting",
                    "Project Management": "Obtain PMP certification or learn Agile/Scrum methodologies",
                    "Math": "Develop quantitative skills including statistics and financial modeling",
                    "Research": "Learn market research techniques and data analysis methods",
                    "Tech-Savvy": "Master business intelligence tools like Tableau, Power BI, Excel"
                }
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
                "improvement_tips": {
                    "Working with People": "Volunteer with children or take child development courses",
                    "Empathy": "Practice active listening and emotional intelligence development",
                    "Public Speaking": "Gain experience presenting to groups and managing classroom dynamics",
                    "Creative": "Develop lesson planning skills and creative teaching methodologies",
                    "Time Management": "Learn classroom management and curriculum planning strategies"
                }
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
                "improvement_tips": {
                    "Empathy": "Volunteer in healthcare settings and practice patient communication",
                    "Working with People": "Develop bedside manner and patient advocacy skills",
                    "Attention to Detail": "Practice medical procedures and medication administration protocols",
                    "Problem Solving": "Study clinical reasoning and critical thinking in healthcare",
                    "Time Management": "Learn to prioritize patient care and manage multiple responsibilities"
                }
            }
        }

        # Course recommendations
        self.course_recommendations = {
            "Programming": {
                "free": [
                    {"name": "freeCodeCamp Full Stack Web Development", "url": "https://www.freecodecamp.org/", "provider": "freeCodeCamp"},
                    {"name": "Codecademy Python Basics", "url": "https://www.codecademy.com/learn/learn-python-3", "provider": "Codecademy"}
                ],
                "paid": [
                    {"name": "Complete Python Bootcamp", "url": "https://www.udemy.com/course/complete-python-bootcamp/", "provider": "Udemy", "price": "$84.99"}
                ]
            },
            "Public Speaking": {
                "free": [
                    {"name": "Introduction to Public Speaking", "url": "https://www.coursera.org/learn/public-speaking", "provider": "University of Washington"},
                    {"name": "Toastmasters International", "url": "https://www.toastmasters.org/", "provider": "Toastmasters"}
                ],
                "paid": [
                    {"name": "The Complete Public Speaking Course", "url": "https://www.udemy.com/course/the-complete-public-speaking-course/", "provider": "Udemy", "price": "$94.99"}
                ]
            },
            "Leadership": {
                "free": [
                    {"name": "Leading People and Teams", "url": "https://www.coursera.org/specializations/leading-teams", "provider": "University of Michigan"}
                ],
                "paid": [
                    {"name": "Complete Leadership Course", "url": "https://www.udemy.com/course/leadership-course/", "provider": "Udemy", "price": "$89.99"}
                ]
            }
        }

        # Interview questions (abbreviated)
        self.interview_questions = {
            "Skills": {
                "Programming": [
                    "Walk me through how you would approach debugging a complex piece of code.",
                    "Describe a challenging technical problem you solved. What was your process?"
                ],
                "Public Speaking": [
                    "Tell me about a time you had to present to a difficult or skeptical audience.",
                    "How do you prepare for important presentations?"
                ]
            },
            "Values": {
                "Impact": [
                    "What kind of work makes you feel most fulfilled?",
                    "Tell me about a project where you felt you made a real difference."
                ]
            }
        }

    def create_profile_from_request(self, request: PersonProfileRequest) -> PersonProfile:
        """Create PersonProfile from API request"""
        # Convert personality traits
        personality = {
            "openness": request.openness,
            "conscientiousness": request.conscientiousness,
            "extraversion": request.extraversion,
            "agreeableness": request.agreeableness,
            "neuroticism": request.neuroticism
        }
        
        # Convert work values (rankings to scores: 1=most important becomes 6, 6=least becomes 1)
        work_values = {
            "income": 7 - request.income_importance,
            "impact": 7 - request.impact_importance,
            "stability": 7 - request.stability_importance,
            "variety": 7 - request.variety_importance,
            "recognition": 7 - request.recognition_importance,
            "autonomy": 7 - request.autonomy_importance
        }
        
        # Skills mapping
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
        
        # Generate interview preparation
        interview_prep = self._generate_interview_preparation(profile, job)
        
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
            "interview_preparation": interview_prep,
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
                courses = self.course_recommendations.get(skill_name, {
                    "free": [{"name": f"Search for free {skill_name} courses", "url": "https://www.coursera.org/", "provider": "Various"}],
                    "paid": [{"name": f"Search for {skill_name} courses", "url": "https://www.udemy.com/", "provider": "Udemy"}]
                })
                
                improvements.append({
                    "skill": skill_name,
                    "current_level": user_level,
                    "required_level": job_requirement,
                    "gap_severity": "High" if job_requirement - user_level > 1.5 else "Medium",
                    "free_courses": courses.get("free", []),
                    "paid_courses": courses.get("paid", []),
                    "improvement_tip": job.get("improvement_tips", {}).get(skill_name, f"Develop {skill_name} skills through practice and training")
                })
        
        return strengths, improvements

    def _generate_interview_preparation(self, profile: PersonProfile, job: Dict) -> Dict:
        """Generate interview preparation based on profile and job match"""
        prep_data = {
            "skills_questions": [],
            "values_questions": [],
            "answer_templates": {}
        }
        
        # Get relevant questions for top skills
        for skill in job["required_skills"][:3]:
            if skill in self.interview_questions["Skills"]:
                prep_data["skills_questions"].extend(self.interview_questions["Skills"][skill][:2])
        
        # Get questions for top work values
        top_values = sorted(profile.work_values.items(), key=lambda x: x[1], reverse=True)[:2]
        for value_name, _ in top_values:
            capitalized_value = value_name.replace('_', ' ').title()
            if capitalized_value in self.interview_questions["Values"]:
                prep_data["values_questions"].extend(self.interview_questions["Values"][capitalized_value][:2])
        
        # Generate answer templates
        strengths_list = [skill for skill in job['required_skills'][:2]]
        values_list = [v.replace('_', ' ').title() for v, _ in top_values[:2]]
        
        prep_data["answer_templates"] = {
            "skills": f"Emphasize your {', '.join(strengths_list)} experience with specific examples.",
            "values": f"Highlight how this role aligns with your core values of {', '.join(values_list)}."
        }
        
        return prep_data

    def analyze_person(self, profile: PersonProfile) -> Dict:
        """Analyze a person against all jobs and return ranked matches"""
        matches = []
        
        for job_name in self.onet_jobs.keys():
            match_result = self.calculate_job_match(profile, job_name)
            matches.append(match_result)
        
        matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        return {
            "profile": asdict(profile),
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Initialize FastAPI app
app = FastAPI(title="Career Matching API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the matcher
matcher = CareerMatcher()

@app.post("/analyze-profile", response_model=AnalysisResponse)
async def analyze_profile(request: PersonProfileRequest):
    """
    Analyze a person's profile and return career matching results
    """
    try:
        # Create profile from request
        profile = matcher.create_profile_from_request(request)
        
        # Analyze the profile
        result = matcher.analyze_person(profile)
        
        return AnalysisResponse(
            success=True,
            message=f"Successfully analyzed profile for {profile.name}",
            result=result,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing profile: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Career Matching API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze-profile": "POST - Analyze individual profile from form data",
            "/jobs": "GET - List available job types",
            "/form-fields": "GET - Get form field specifications",
            "/health": "GET - Health check"
        }
    }

@app.get("/form-fields")
async def get_form_fields():
    """Get form field specifications for frontend"""
    return {
        "personality_traits": {
            "openness": {"type": "integer", "min": 1, "max": 5, "description": "Openness to experience"},
            "conscientiousness": {"type": "integer", "min": 1, "max": 5, "description": "Conscientiousness"},
            "extraversion": {"type": "integer", "min": 1, "max": 5, "description": "Extraversion"},
            "agreeableness": {"type": "integer", "min": 1, "max": 5, "description": "Agreeableness"},
            "neuroticism": {"type": "integer", "min": 1, "max": 5, "description": "Neuroticism"}
        },
        "work_values": {
            "income_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of income (1=most important)"},
            "impact_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of making an impact"},
            "stability_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of job stability"},
            "variety_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of variety in work"},
            "recognition_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of recognition"},
            "autonomy_importance": {"type": "integer", "min": 1, "max": 6, "description": "Importance of autonomy"}
        },
        "skills": {
            "math": {"type": "integer", "min": 1, "max": 5, "description": "Math skills"},
            "problem_solving": {"type": "integer", "min": 1, "max": 5, "description": "Problem solving skills"},
            "public_speaking": {"type": "integer", "min": 1, "max": 5, "description": "Public speaking skills"},
            "creative": {"type": "integer", "min": 1, "max": 5, "description": "Creative skills"},
            "working_with_people": {"type": "integer", "min": 1, "max": 5, "description": "People skills"},
            "writing": {"type": "integer", "min": 1, "max": 5, "description": "Writing & communication"},
            "tech_savvy": {"type": "integer", "min": 1, "max": 5, "description": "Technology skills"},
            "leadership": {"type": "integer", "min": 1, "max": 5, "description": "Leadership & management"},
            "networking": {"type": "integer", "min": 1, "max": 5, "description": "Networking & relationship building"},
            "programming": {"type": "integer", "min": 1, "max": 5, "description": "Programming proficiency"},
            "empathy": {"type": "integer", "min": 1, "max": 5, "description": "Empathy"},
            "time_management": {"type": "integer", "min": 1, "max": 5, "description": "Time management"},
            "attention_to_detail": {"type": "integer", "min": 1, "max": 5, "description": "Attention to detail"},
            "project_management": {"type": "integer", "min": 1, "max": 5, "description": "Project management"},
            "research": {"type": "integer", "min": 1, "max": 5, "description": "Research skills"},
            "teamwork": {"type": "integer", "min": 1, "max": 5, "description": "Teamwork"}
        },
        "interests": {
            "options": ["investigative", "social", "artistic", "enterprising", "realistic", "conventional"],
            "type": "array",
            "description": "Select multiple interests that apply"
        },
        "basic_info": {
            "name": {"type": "string", "required": True, "description": "Full name"},
            "email": {"type": "string", "required": True, "description": "Email address"},
            "university": {"type": "string", "required": True, "description": "University or institution"},
            "preferred_career": {"type": "string", "required": True, "description": "Preferred career path"}
        }
    }

@app.get("/jobs")
async def get_jobs():
    """Get list of available job types and their details"""
    jobs_info = {}
    for job_name, job_data in matcher.onet_jobs.items():
        jobs_info[job_name] = {
            "onet_code": job_data["onet_code"],
            "required_skills": job_data["required_skills"],
            "top_work_values": sorted(job_data["work_values"].items(), key=lambda x: x[1], reverse=True)[:3],
            "top_interests": sorted(job_data["interests"].items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    return {
        "available_jobs": list(matcher.onet_jobs.keys()),
        "job_details": jobs_info
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Example HTML form endpoint (optional - for testing)
@app.get("/form")
async def get_form():
    """Return a simple HTML form for testing the API"""
    html_form = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Career Matching Form</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-group { margin: 15px 0; }
            label { display: block; font-weight: bold; margin-bottom: 5px; }
            input, select, textarea { width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .section { border: 1px solid #eee; padding: 15px; margin: 20px 0; border-radius: 8px; }
            .checkbox-group { display: flex; flex-wrap: wrap; gap: 15px; }
            .checkbox-item { display: flex; align-items: center; gap: 5px; }
        </style>
    </head>
    <body>
        <h1>Career Matching Assessment</h1>
        <form id="careerForm">
            <div class="section">
                <h2>Basic Information</h2>
                <div class="form-group">
                    <label for="name">Full Name:</label>
                    <input type="text" id="name" name="name" required>
                </div>
                <div class="form-group">
                    <label for="email">Email:</label>
                    <input type="email" id="email" name="email" required>
                </div>
                <div class="form-group">
                    <label for="university">University:</label>
                    <input type="text" id="university" name="university" required>
                </div>
                <div class="form-group">
                    <label for="preferred_career">Preferred Career:</label>
                    <input type="text" id="preferred_career" name="preferred_career" required>
                </div>
            </div>

            <div class="section">
                <h2>Personality Traits (1-5 scale)</h2>
                <div class="form-group">
                    <label for="openness">Openness to Experience (1=Low, 5=High):</label>
                    <input type="range" id="openness" name="openness" min="1" max="5" value="3">
                    <span id="openness-value">3</span>
                </div>
                <div class="form-group">
                    <label for="conscientiousness">Conscientiousness (1=Low, 5=High):</label>
                    <input type="range" id="conscientiousness" name="conscientiousness" min="1" max="5" value="3">
                    <span id="conscientiousness-value">3</span>
                </div>
                <div class="form-group">
                    <label for="extraversion">Extraversion (1=Low, 5=High):</label>
                    <input type="range" id="extraversion" name="extraversion" min="1" max="5" value="3">
                    <span id="extraversion-value">3</span>
                </div>
                <div class="form-group">
                    <label for="agreeableness">Agreeableness (1=Low, 5=High):</label>
                    <input type="range" id="agreeableness" name="agreeableness" min="1" max="5" value="3">
                    <span id="agreeableness-value">3</span>
                </div>
                <div class="form-group">
                    <label for="neuroticism">Neuroticism (1=Low, 5=High):</label>
                    <input type="range" id="neuroticism" name="neuroticism" min="1" max="5" value="3">
                    <span id="neuroticism-value">3</span>
                </div>
            </div>

            <div class="section">
                <h2>Work Values Importance (1=Most Important, 6=Least Important)</h2>
                <div class="form-group">
                    <label for="income_importance">Income Importance:</label>
                    <select id="income_importance" name="income_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="impact_importance">Impact Importance:</label>
                    <select id="impact_importance" name="impact_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="stability_importance">Stability Importance:</label>
                    <select id="stability_importance" name="stability_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="variety_importance">Variety Importance:</label>
                    <select id="variety_importance" name="variety_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="recognition_importance">Recognition Importance:</label>
                    <select id="recognition_importance" name="recognition_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="autonomy_importance">Autonomy Importance:</label>
                    <select id="autonomy_importance" name="autonomy_importance" required>
                        <option value="1">1 - Most Important</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6 - Least Important</option>
                    </select>
                </div>
            </div>

            <div class="section">
                <h2>Skills Assessment (1=Poor, 5=Excellent)</h2>
                <div class="form-group">
                    <label for="math">Math Skills:</label>
                    <input type="range" id="math" name="math" min="1" max="5" value="3">
                    <span id="math-value">3</span>
                </div>
                <div class="form-group">
                    <label for="problem_solving">Problem Solving:</label>
                    <input type="range" id="problem_solving" name="problem_solving" min="1" max="5" value="3">
                    <span id="problem_solving-value">3</span>
                </div>
                <div class="form-group">
                    <label for="public_speaking">Public Speaking:</label>
                    <input type="range" id="public_speaking" name="public_speaking" min="1" max="5" value="3">
                    <span id="public_speaking-value">3</span>
                </div>
                <div class="form-group">
                    <label for="creative">Creativity:</label>
                    <input type="range" id="creative" name="creative" min="1" max="5" value="3">
                    <span id="creative-value">3</span>
                </div>
                <div class="form-group">
                    <label for="working_with_people">Working with People:</label>
                    <input type="range" id="working_with_people" name="working_with_people" min="1" max="5" value="3">
                    <span id="working_with_people-value">3</span>
                </div>
                <div class="form-group">
                    <label for="writing">Writing & Communication:</label>
                    <input type="range" id="writing" name="writing" min="1" max="5" value="3">
                    <span id="writing-value">3</span>
                </div>
                <div class="form-group">
                    <label for="tech_savvy">Technology Skills:</label>
                    <input type="range" id="tech_savvy" name="tech_savvy" min="1" max="5" value="3">
                    <span id="tech_savvy-value">3</span>
                </div>
                <div class="form-group">
                    <label for="leadership">Leadership:</label>
                    <input type="range" id="leadership" name="leadership" min="1" max="5" value="3">
                    <span id="leadership-value">3</span>
                </div>
                <div class="form-group">
                    <label for="networking">Networking:</label>
                    <input type="range" id="networking" name="networking" min="1" max="5" value="3">
                    <span id="networking-value">3</span>
                </div>
                <div class="form-group">
                    <label for="programming">Programming:</label>
                    <input type="range" id="programming" name="programming" min="1" max="5" value="3">
                    <span id="programming-value">3</span>
                </div>
                <div class="form-group">
                    <label for="empathy">Empathy:</label>
                    <input type="range" id="empathy" name="empathy" min="1" max="5" value="3">
                    <span id="empathy-value">3</span>
                </div>
                <div class="form-group">
                    <label for="time_management">Time Management:</label>
                    <input type="range" id="time_management" name="time_management" min="1" max="5" value="3">
                    <span id="time_management-value">3</span>
                </div>
                <div class="form-group">
                    <label for="attention_to_detail">Attention to Detail:</label>
                    <input type="range" id="attention_to_detail" name="attention_to_detail" min="1" max="5" value="3">
                    <span id="attention_to_detail-value">3</span>
                </div>
                <div class="form-group">
                    <label for="project_management">Project Management:</label>
                    <input type="range" id="project_management" name="project_management" min="1" max="5" value="3">
                    <span id="project_management-value">3</span>
                </div>
                <div class="form-group">
                    <label for="research">Research Skills:</label>
                    <input type="range" id="research" name="research" min="1" max="5" value="3">
                    <span id="research-value">3</span>
                </div>
                <div class="form-group">
                    <label for="teamwork">Teamwork:</label>
                    <input type="range" id="teamwork" name="teamwork" min="1" max="5" value="3">
                    <span id="teamwork-value">3</span>
                </div>
            </div>

            <div class="section">
                <h2>Career Interests</h2>
                <p>Select all that apply:</p>
                <div class="checkbox-group">
                    <div class="checkbox-item">
                        <input type="checkbox" id="investigative" name="interests" value="investigative">
                        <label for="investigative">Investigative (research, analysis)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="social" name="interests" value="social">
                        <label for="social">Social (helping, teaching)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="artistic" name="interests" value="artistic">
                        <label for="artistic">Artistic (creative, expressive)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="enterprising" name="interests" value="enterprising">
                        <label for="enterprising">Enterprising (leading, selling)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="realistic" name="interests" value="realistic">
                        <label for="realistic">Realistic (hands-on, practical)</label>
                    </div>
                    <div class="checkbox-item">
                        <input type="checkbox" id="conventional" name="interests" value="conventional">
                        <label for="conventional">Conventional (organized, detail-oriented)</label>
                    </div>
                </div>
            </div>

            <button type="submit">Analyze Career Match</button>
        </form>

        <div id="results" style="display: none;">
            <h2>Analysis Results</h2>
            <pre id="results-content"></pre>
        </div>

        <script>
            // Update slider values
            document.querySelectorAll('input[type="range"]').forEach(slider => {
                const valueSpan = document.getElementById(slider.id + '-value');
                slider.addEventListener('input', () => {
                    valueSpan.textContent = slider.value;
                });
            });

            // Handle form submission
            document.getElementById('careerForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {};
                
                // Get basic form data
                for (const [key, value] of formData.entries()) {
                    if (key !== 'interests') {
                        data[key] = key.includes('importance') ? parseInt(value) : 
                                   (key === 'name' || key === 'email' || key === 'university' || key === 'preferred_career') ? value : 
                                   parseInt(value);
                    }
                }
                
                // Get interests array
                data.interests = Array.from(formData.getAll('interests'));
                
                try {
                    const response = await fetch('/analyze-profile', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    document.getElementById('results-content').textContent = JSON.stringify(result, null, 2);
                    document.getElementById('results').style.display = 'block';
                    document.getElementById('results').scrollIntoView();
                } catch (error) {
                    alert('Error analyzing profile: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_form, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)