import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

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
        # O*NET Job Database with detailed metrics including Work Styles
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
                    "leadership": 4.1, "writing": 3.9, "negotiation": 4.0
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
                "required_skills": ["Public Speaking", "Creative", "Networking & Relationship Building", "Leadership & Team Management"],
                "improvement_tips": {
                    "Public Speaking": "Join Toastmasters or take presentation skills workshops",
                    "Creative": "Develop creative thinking through design courses and brainstorming techniques",
                    "Networking & Relationship Building": "Attend industry events and practice relationship-building strategies",
                    "Leadership & Team Management": "Seek leadership opportunities and consider management training"
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
        
        # Skill mappings from survey to standardized names
        self.skill_mapping = {
            "Math": "math",
            "Problem Solving": "problem_solving",
            "Public Speaking": "public_speaking",
            "Creative": "creative",
            "Working with People": "working_with_people",
            "Writing & Communication": "writing",
            "Tech-Savvy (Excel, Tableau, etc.)": "tech_savvy",
            "Leadership & Team Management": "leadership",
            "Networking & Relationship Building": "networking",
            "Negotiation & Salesmanship": "negotiation",
            "Creativity & Innovation": "creative",
            "Programming Proficiency": "programming",
            "Languages": "languages",
            "Empathy": "empathy",
            "Time Management": "time_management",
            "Attention to Detail": "attention_to_detail",
            "Project Management": "project_management",
            "Artistic": "artistic",
            "Research": "research",
            "Hands on Building & Prototyping": "hands_on",
            "Teamwork": "teamwork"
        }

        # Course recommendations with actual links
        self.course_recommendations = {
            "Programming": {
                "free": [
                    {"name": "freeCodeCamp Full Stack Web Development", "url": "https://www.freecodecamp.org/", "provider": "freeCodeCamp"},
                    {"name": "Codecademy Python Basics", "url": "https://www.codecademy.com/learn/learn-python-3", "provider": "Codecademy"},
                    {"name": "CS50 Introduction to Computer Science", "url": "https://cs50.harvard.edu/x/", "provider": "Harvard"}
                ],
                "paid": [
                    {"name": "Complete Python Bootcamp", "url": "https://www.udemy.com/course/complete-python-bootcamp/", "provider": "Udemy", "price": "$84.99"},
                    {"name": "Python for Everybody Specialization", "url": "https://www.coursera.org/specializations/python", "provider": "Coursera", "price": "$49/month"}
                ]
            },
            "Public Speaking": {
                "free": [
                    {"name": "Introduction to Public Speaking", "url": "https://www.coursera.org/learn/public-speaking", "provider": "University of Washington"},
                    {"name": "Toastmasters International", "url": "https://www.toastmasters.org/", "provider": "Toastmasters"},
                    {"name": "TED Masterclass - Public Speaking", "url": "https://www.ted.com/participate/ted-masterclass", "provider": "TED"}
                ],
                "paid": [
                    {"name": "The Complete Public Speaking Course", "url": "https://www.udemy.com/course/the-complete-public-speaking-course/", "provider": "Udemy", "price": "$94.99"},
                    {"name": "Speaking to Inspire", "url": "https://www.masterclass.com/classes/chris-anderson-public-speaking", "provider": "MasterClass", "price": "$180/year"}
                ]
            },
            "Leadership & Team Management": {
                "free": [
                    {"name": "Leading People and Teams", "url": "https://www.coursera.org/specializations/leading-teams", "provider": "University of Michigan"},
                    {"name": "Management Fundamentals", "url": "https://www.edx.org/course/management-fundamentals", "provider": "edX"}
                ],
                "paid": [
                    {"name": "Complete Leadership Course", "url": "https://www.udemy.com/course/leadership-course/", "provider": "Udemy", "price": "$89.99"},
                    {"name": "Strategic Leadership Certificate", "url": "https://www.coursera.org/professional-certificates/strategic-leadership", "provider": "University of Virginia", "price": "$59/month"}
                ]
            },
            "Project Management": {
                "free": [
                    {"name": "Google Project Management Certificate", "url": "https://www.coursera.org/professional-certificates/google-project-management", "provider": "Google/Coursera"},
                    {"name": "Project Management Fundamentals", "url": "https://www.edx.org/course/project-management-fundamentals", "provider": "edX"}
                ],
                "paid": [
                    {"name": "PMP Certification Training", "url": "https://www.udemy.com/course/pmp-certification-exam-prep-course-pmbok-6th-edition/", "provider": "Udemy", "price": "$149.99"},
                    {"name": "Agile Project Management", "url": "https://www.coursera.org/specializations/agile-project-management", "provider": "Google", "price": "$49/month"}
                ]
            },
            "Creative": {
                "free": [
                    {"name": "Creative Thinking Course", "url": "https://www.edx.org/course/creative-thinking", "provider": "edX"},
                    {"name": "Design Thinking Course", "url": "https://www.coursera.org/learn/design-thinking-innovation", "provider": "University of Virginia"}
                ],
                "paid": [
                    {"name": "Creativity and Innovation", "url": "https://www.udemy.com/course/creativity-and-innovation/", "provider": "Udemy", "price": "$74.99"},
                    {"name": "Creative Problem Solving", "url": "https://www.skillshare.com/classes/Creative-Problem-Solving/", "provider": "Skillshare", "price": "$168/year"}
                ]
            },
            "Math": {
                "free": [
                    {"name": "Khan Academy Mathematics", "url": "https://www.khanacademy.org/math", "provider": "Khan Academy"},
                    {"name": "MIT Calculus Course", "url": "https://ocw.mit.edu/courses/mathematics/", "provider": "MIT OpenCourseWare"}
                ],
                "paid": [
                    {"name": "Complete Mathematics Course", "url": "https://www.udemy.com/course/complete-mathematics-course/", "provider": "Udemy", "price": "$89.99"},
                    {"name": "Statistics and Probability", "url": "https://www.coursera.org/specializations/statistics-probability", "provider": "Duke University", "price": "$49/month"}
                ]
            },
            "Problem Solving": {
                "free": [
                    {"name": "Algorithmic Thinking", "url": "https://www.coursera.org/learn/algorithmic-thinking-1", "provider": "Rice University"},
                    {"name": "Critical Thinking", "url": "https://www.edx.org/course/critical-thinking", "provider": "edX"}
                ],
                "paid": [
                    {"name": "Problem Solving Techniques", "url": "https://www.udemy.com/course/problem-solving-techniques/", "provider": "Udemy", "price": "$79.99"},
                    {"name": "Systems Thinking", "url": "https://www.coursera.org/learn/systems-thinking", "provider": "University of Virginia", "price": "$49/month"}
                ]
            },
            "Working with People": {
                "free": [
                    {"name": "Interpersonal Skills", "url": "https://www.coursera.org/learn/interpersonal-communication", "provider": "University of California Davis"},
                    {"name": "Emotional Intelligence", "url": "https://www.edx.org/course/emotional-intelligence", "provider": "edX"}
                ],
                "paid": [
                    {"name": "People Skills Mastery", "url": "https://www.udemy.com/course/people-skills/", "provider": "Udemy", "price": "$84.99"},
                    {"name": "Social Psychology", "url": "https://www.coursera.org/learn/social-psychology", "provider": "Wesleyan University", "price": "$49/month"}
                ]
            },
            "Empathy": {
                "free": [
                    {"name": "Empathy Course", "url": "https://www.futurelearn.com/courses/empathy-compassion", "provider": "FutureLearn"},
                    {"name": "Emotional Intelligence Fundamentals", "url": "https://www.edx.org/course/emotional-intelligence-fundamentals", "provider": "edX"}
                ],
                "paid": [
                    {"name": "Emotional Intelligence Mastery", "url": "https://www.udemy.com/course/emotional-intelligence-mastery/", "provider": "Udemy", "price": "$79.99"},
                    {"name": "Compassion Cultivation", "url": "https://www.mindful.org/online-courses/", "provider": "Mindful.org", "price": "$297"}
                ]
            }
        }

        # Interview questions database
        self.interview_questions = {
            "Skills": {
                "Programming": [
                    "Walk me through how you would approach debugging a complex piece of code.",
                    "Describe a challenging technical problem you solved. What was your process?",
                    "How do you stay current with new programming languages and technologies?",
                    "Tell me about a time you had to learn a new programming language or framework quickly."
                ],
                "Public Speaking": [
                    "Tell me about a time you had to present to a difficult or skeptical audience.",
                    "How do you prepare for important presentations?",
                    "Describe a situation where you had to explain a complex topic to non-experts.",
                    "What techniques do you use to engage your audience during presentations?"
                ],
                "Problem Solving": [
                    "Describe your approach to solving complex problems with no clear solution.",
                    "Tell me about a time you had to think outside the box to solve a problem.",
                    "How do you prioritize when facing multiple urgent problems?",
                    "Walk me through a recent problem you solved step by step."
                ],
                "Leadership & Team Management": [
                    "Tell me about a time you had to lead a team through a difficult situation.",
                    "How do you handle conflict within your team?",
                    "Describe your leadership style and give me an example of it in action.",
                    "Tell me about a time you had to give difficult feedback to a team member."
                ],
                "Creative": [
                    "Tell me about a time you came up with a creative solution to a problem.",
                    "How do you foster creativity in your work or team?",
                    "Describe a project where you had to think creatively under constraints.",
                    "What's the most innovative idea you've implemented?"
                ],
                "Working with People": [
                    "Tell me about a time you had to work with a difficult colleague or client.",
                    "How do you build rapport with new team members or clients?",
                    "Describe a situation where you had to collaborate with people from different backgrounds.",
                    "Tell me about a time you had to influence someone without authority."
                ],
                "Empathy": [
                    "Tell me about a time you helped someone who was struggling.",
                    "How do you handle situations where someone is upset or emotional?",
                    "Describe a time you had to see a situation from someone else's perspective.",
                    "Tell me about a time you went above and beyond to help a colleague or client."
                ]
            },
            "Values": {
                "Impact": [
                    "What kind of work makes you feel most fulfilled?",
                    "Tell me about a project where you felt you made a real difference.",
                    "How important is it for you to see the direct results of your work?",
                    "Describe a time when your work had a meaningful impact on others."
                ],
                "Autonomy": [
                    "Describe your ideal work environment in terms of supervision and independence.",
                    "How do you handle situations where you disagree with management decisions?",
                    "Tell me about a time you took initiative without being asked.",
                    "What level of guidance do you prefer when starting a new project?"
                ],
                "Income": [
                    "How important is financial compensation in your career decisions?",
                    "Tell me about a time you had to balance salary expectations with other job factors.",
                    "How do you define fair compensation for your work?",
                    "What motivates you more: money or job satisfaction?"
                ],
                "Stability": [
                    "How important is job security to you?",
                    "Tell me about a time you had to adapt to major organizational changes.",
                    "What does work-life balance mean to you?",
                    "How do you handle uncertainty in the workplace?"
                ],
                "Recognition": [
                    "How important is it for you to receive recognition for your work?",
                    "Tell me about a time you felt your contributions weren't acknowledged.",
                    "What kind of recognition motivates you most?",
                    "How do you prefer to celebrate achievements with your team?"
                ],
                "Variety": [
                    "Do you prefer routine tasks or varied responsibilities?",
                    "Tell me about a time you had to handle multiple different types of projects.",
                    "How do you stay motivated when doing repetitive work?",
                    "What keeps you engaged in your work day-to-day?"
                ]
            },
            "Interests": {
                "Investigative": [
                    "What kind of problems do you enjoy solving most?",
                    "Tell me about a time you had to research something complex.",
                    "How do you approach learning new concepts or skills?",
                    "Describe a time you had to analyze data to make a decision."
                ],
                "Social": [
                    "Describe a time when you helped someone overcome a challenge.",
                    "What energizes you more: working alone or with others?",
                    "Tell me about your experience mentoring or teaching others.",
                    "How do you handle interpersonal conflicts?"
                ],
                "Artistic": [
                    "Tell me about a time you brought creativity to your work.",
                    "How do you express your creative side?",
                    "Describe a project where aesthetics or design were important.",
                    "What role does creativity play in your ideal job?"
                ],
                "Enterprising": [
                    "Tell me about a time you identified a new business opportunity.",
                    "How comfortable are you with taking calculated risks?",
                    "Describe your experience with sales or persuasion.",
                    "Tell me about a time you had to influence others to adopt your ideas."
                ],
                "Realistic": [
                    "Do you prefer working with your hands or working with ideas?",
                    "Tell me about a time you had to build or fix something.",
                    "How comfortable are you with tools and equipment?",
                    "Describe a hands-on project you've completed."
                ],
                "Conventional": [
                    "How do you organize your work and maintain attention to detail?",
                    "Tell me about a time you had to follow strict procedures or guidelines.",
                    "How do you ensure accuracy in your work?",
                    "Describe your approach to managing data or records."
                ]
            },
            "Work Styles": {
                "Analytical Thinking": [
                    "Walk me through your decision-making process for complex problems.",
                    "Tell me about a time you had to analyze data to make a recommendation.",
                    "How do you break down large, complex projects?",
                    "Describe a situation where you had to think systematically about a problem."
                ],
                "Adaptability": [
                    "Tell me about a time when priorities changed suddenly. How did you handle it?",
                    "Describe a situation where you had to learn something completely new.",
                    "How do you handle working in ambiguous or changing environments?",
                    "Tell me about a time you had to adjust your approach mid-project."
                ],
                "Stress Tolerance": [
                    "How do you handle high-pressure situations?",
                    "Tell me about a time you had to work under a tight deadline.",
                    "Describe your strategies for managing stress at work.",
                    "Tell me about a challenging period at work and how you got through it."
                ],
                "Dependability": [
                    "How do you ensure you meet all your commitments and deadlines?",
                    "Tell me about a time you had to be relied upon in a critical situation.",
                    "Describe your approach to managing multiple responsibilities.",
                    "How do you maintain consistency in your work quality?"
                ],
                "Leadership": [
                    "Tell me about a time you had to step up and lead without being asked.",
                    "How do you motivate others to achieve their best work?",
                    "Describe a situation where you had to make a difficult leadership decision.",
                    "Tell me about your approach to developing others."
                ],
                "Initiative": [
                    "Tell me about a time you identified and solved a problem proactively.",
                    "Describe a situation where you went beyond what was expected.",
                    "How do you identify opportunities for improvement in your work?",
                    "Tell me about a time you started a new project or process."
                ]
            }
        }

    def parse_csv_row(self, row: pd.Series) -> PersonProfile:
        """Parse a CSV row into a PersonProfile object"""
        try:
            # Basic info
            name = row.iloc[1] if pd.notna(row.iloc[1]) else "Unknown"
            email = row.iloc[2] if pd.notna(row.iloc[2]) else "Unknown"
            university = row.iloc[3] if pd.notna(row.iloc[3]) else "Unknown"
            
            # Personality traits (Big 5 approximation)
            personality = {
                "openness": (self._safe_int(row.iloc[4]) + self._safe_int(row.iloc[5])) / 2,
                "conscientiousness": (self._safe_int(row.iloc[6]) + self._safe_int(row.iloc[7])) / 2,
                "extraversion": (self._safe_int(row.iloc[8]) + self._safe_int(row.iloc[9])) / 2,
                "agreeableness": (self._safe_int(row.iloc[10]) + self._safe_int(row.iloc[11])) / 2,
                "neuroticism": (self._safe_int(row.iloc[12]) + self._safe_int(row.iloc[13])) / 2
            }
            
            # Work values (convert rankings to scores: 1=most important becomes 6, 6=least becomes 1)
            work_values = {
                "income": 7 - self._safe_int(row.iloc[14]),
                "impact": 7 - self._safe_int(row.iloc[15]),
                "stability": 7 - self._safe_int(row.iloc[16]),
                "variety": 7 - self._safe_int(row.iloc[17]),
                "recognition": 7 - self._safe_int(row.iloc[18]),
                "autonomy": 7 - self._safe_int(row.iloc[19])
            }
            
            # Skills (columns 22-41 based on the sample data)
            skills = {}
            skill_columns = [
                "Math", "Problem Solving", "Public Speaking", "Creative", "Working with People",
                "Writing & Communication", "Tech-Savvy (Excel, Tableau, etc.)", "Leadership & Team Management",
                "Networking & Relationship Building", "Negotiation & Salesmanship", "Creativity & Innovation",
                "Programming Proficiency", "Languages", "Empathy", "Time Management", "Attention to Detail",
                "Project Management", "Artistic", "Research", "Hands on Building & Prototyping", "Teamwork"
            ]
            
            for i, skill in enumerate(skill_columns, start=22):
                if i < len(row) and pd.notna(row.iloc[i]):
                    mapped_skill = self.skill_mapping.get(skill, skill.lower().replace(' ', '_'))
                    skills[mapped_skill] = self._safe_int(row.iloc[i])
            
            # Interests and preferred career
            interests = []
            if len(row) > 43 and pd.notna(row.iloc[43]):
                interests_text = str(row.iloc[43])
                if "Investigative" in interests_text:
                    interests.append("investigative")
                if "Enterprising" in interests_text:
                    interests.append("enterprising")
                if "Social" in interests_text:
                    interests.append("social")
                if "Artistic" in interests_text:
                    interests.append("artistic")
                if "Realistic" in interests_text:
                    interests.append("realistic")
                if "Conventional" in interests_text:
                    interests.append("conventional")
            
            preferred_career = str(row.iloc[44]) if len(row) > 44 and pd.notna(row.iloc[44]) else "Not specified"
            
            return PersonProfile(
                name=name,
                email=email,
                university=university,
                personality=personality,
                work_values=work_values,
                skills=skills,
                interests=interests,
                preferred_career=preferred_career
            )
            
        except Exception as e:
            print(f"Error parsing row: {e}")
            return None

    def _safe_int(self, value) -> int:
        """Safely convert value to int, handling various formats"""
        if pd.isna(value):
            return 3  # Default neutral value
        
        try:
            # Handle string values like "5 - Strongly Agree"
            if isinstance(value, str):
                # Extract first number
                import re
                numbers = re.findall(r'\d+', value)
                if numbers:
                    return int(numbers[0])
                return 3
            return int(float(value))
        except:
            return 3

    def calculate_job_match(self, profile: PersonProfile, job_name: str) -> Dict:
        """Calculate comprehensive match percentage and details for a specific job"""
        job = self.onet_jobs[job_name]
        
        # Skills match (30% weight)
        skills_score = self._calculate_skills_match(profile.skills, job["skills"])
        
        # Work values match (25% weight)
        values_score = self._calculate_values_match(profile.work_values, job["work_values"])
        
        # Interests match (20% weight)
        interests_score = self._calculate_interests_match(profile.interests, job["interests"])
        
        # Work styles match (25% weight) - NEW
        work_styles_score = self._calculate_work_styles_match(profile.personality, job.get("work_styles", {}))
        
        # Overall match
        overall_match = (skills_score * 0.3 + values_score * 0.25 + 
                        interests_score * 0.2 + work_styles_score * 0.25) * 100
        
        # Identify strengths and improvement areas with course recommendations
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
            user_level = user_skills.get(skill, 2.5)  # Default if skill not found
            
            # Normalize both to 0-1 scale
            normalized_importance = importance / 5.0
            normalized_user = user_level / 5.0
            
            # Calculate match (penalize being too far below requirement)
            if normalized_user >= normalized_importance:
                score = 1.0
            else:
                # Penalty increases as gap widens
                gap = normalized_importance - normalized_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * normalized_importance)  # Weight by importance
        
        return sum(scores) / sum(job_skills.values()) * 5 if scores else 0

    def _calculate_values_match(self, user_values: Dict, job_values: Dict) -> float:
        """Calculate work values compatibility"""
        scores = []
        
        for value, job_importance in job_values.items():
            user_importance = user_values.get(value, 3.5)
            
            # Normalize to 0-1
            norm_job = job_importance / 5.0
            norm_user = user_importance / 6.0  # User values are 1-6 scale
            
            # Calculate similarity (closer values = higher score)
            difference = abs(norm_job - norm_user)
            similarity = 1 - difference
            
            scores.append(similarity * norm_job)  # Weight by job importance
        
        return sum(scores) / sum(job_values.values()) * 5 if scores else 0

    def _calculate_interests_match(self, user_interests: List[str], job_interests: Dict) -> float:
        """Calculate interests compatibility"""
        if not user_interests:
            return 0.5  # Neutral if no interests specified
        
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
            "stress_tolerance": 5 - personality.get("neuroticism", 3),  # Inverse
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
            
            # Normalize and calculate match
            norm_importance = importance / 5.0
            norm_user = user_level / 5.0
            
            # Calculate compatibility
            if norm_user >= norm_importance * 0.8:  # 80% threshold
                score = 1.0
            else:
                gap = (norm_importance * 0.8) - norm_user
                score = max(0, 1 - (gap * 2))
            
            scores.append(score * norm_importance)
        
        return sum(scores) / sum(job_work_styles.values()) * 5 if scores else 0

    def _identify_strengths_improvements(self, profile: PersonProfile, job: Dict) -> Tuple[List[str], List[Dict]]:
        """Identify user strengths and areas for improvement with course recommendations"""
        strengths = []
        improvements = []
        
        # Check skills against job requirements
        for skill_name in job["required_skills"]:
            skill_key = self.skill_mapping.get(skill_name, skill_name.lower().replace(' ', '_'))
            user_level = profile.skills.get(skill_key, 2.5)
            job_requirement = job["skills"].get(skill_key, 3.5)
            
            if user_level >= job_requirement:
                strengths.append(f"Strong {skill_name} skills (Level {user_level}/5)")
            elif user_level < job_requirement - 0.5:
                # Get course recommendations
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
            "interests_questions": [],
            "work_styles_questions": [],
            "answer_templates": {}
        }
        
        # Get relevant questions for top skills
        for skill in job["required_skills"][:3]:  # Top 3 skills
            if skill in self.interview_questions["Skills"]:
                prep_data["skills_questions"].extend(self.interview_questions["Skills"][skill][:2])
        
        # Get questions for top work values
        top_values = sorted(profile.work_values.items(), key=lambda x: x[1], reverse=True)[:2]
        for value_name, _ in top_values:
            capitalized_value = value_name.replace('_', ' ').title()
            if capitalized_value in self.interview_questions["Values"]:
                prep_data["values_questions"].extend(self.interview_questions["Values"][capitalized_value][:2])
        
        # Get questions for interests
        for interest in profile.interests[:2]:
            capitalized_interest = interest.capitalize()
            if capitalized_interest in self.interview_questions["Interests"]:
                prep_data["interests_questions"].extend(self.interview_questions["Interests"][capitalized_interest][:2])
        
        # Get work styles questions based on job requirements
        job_work_styles = job.get("work_styles", {})
        top_work_styles = sorted(job_work_styles.items(), key=lambda x: x[1], reverse=True)[:2]
        for style_name, _ in top_work_styles:
            capitalized_style = style_name.replace('_', ' ').title()
            if capitalized_style in self.interview_questions["Work Styles"]:
                prep_data["work_styles_questions"].extend(self.interview_questions["Work Styles"][capitalized_style][:2])
        
        # Generate answer templates
        strengths_list = [skill.split(' skills')[0].replace('Strong ', '') for skill in job['required_skills'][:2]]
        values_list = [v.replace('_', ' ').title() for v, _ in top_values[:2]]
        
        prep_data["answer_templates"] = {
            "skills": f"Based on your analysis, {profile.name} should emphasize their {', '.join(strengths_list)} experience with specific examples and quantifiable results.",
            "values": f"Highlight how this role aligns with your core values of {', '.join(values_list)}, and provide examples of when these values guided your decisions.",
            "interests": f"Connect your {', '.join(profile.interests[:2])} interests to specific aspects of this role and explain how they drive your motivation.",
            "work_styles": f"Demonstrate your {', '.join([s.replace('_', ' ') for s, _ in top_work_styles])} through concrete examples from your experience."
        }
        
        return prep_data

    def analyze_person(self, profile: PersonProfile) -> Dict:
        """Analyze a person against all jobs and return ranked matches"""
        matches = []
        
        for job_name in self.onet_jobs.keys():
            match_result = self.calculate_job_match(profile, job_name)
            matches.append(match_result)
        
        # Sort by overall match percentage
        matches.sort(key=lambda x: x["overall_match"], reverse=True)
        
        return {
            "profile": profile,
            "matches": matches,
            "top_match": matches[0] if matches else None,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def process_csv_file(self, csv_path: str) -> List[Dict]:
        """Process entire CSV file and return analysis for all people"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            results = []
            
            for index, row in df.iterrows():
                if index == 0:  # Skip header if it's data
                    continue
                    
                profile = self.parse_csv_row(row)
                if profile and profile.name != "Unknown":
                    print(f"Processing: {profile.name}")
                    analysis = self.analyze_person(profile)
                    results.append(analysis)
            
            return results
            
        except Exception as e:
            print(f"Error processing CSV file: {e}")
            return []

    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive MVP report with all required sections"""
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE CAREER MATCHING ANALYSIS REPORT - MVP VERSION")
        report.append("=" * 100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total profiles analyzed: {len(results)}")
        report.append("")
        
        for i, result in enumerate(results, 1):
            profile = result["profile"]
            top_match = result["matches"][0]
            
            report.append(f"{'=' * 80}")
            report.append(f"PROFILE {i}: {profile.name.upper()}")
            report.append(f"{'=' * 80}")
            report.append(f"Email: {profile.email}")
            report.append(f"University: {profile.university}")
            report.append(f"Preferred Career: {profile.preferred_career}")
            report.append("")
            
            # MVP REQUIREMENT 1 & 2: Comparison by Skills, Values, Interests, Work Styles with % scores
            report.append("📊 DETAILED MATCHING BREAKDOWN BY CATEGORY:")
            report.append("-" * 60)
            breakdown = top_match["breakdown"]
            report.append(f"💪 Skills Match:        {breakdown['skills_match']}% - Technical and professional competencies")
            report.append(f"❤️  Values Match:        {breakdown['values_match']}% - Work motivation and priorities")  
            report.append(f"🎯 Interests Match:     {breakdown['interests_match']}% - Natural curiosities and preferences")
            report.append(f"⚙️  Work Styles Match:   {breakdown['work_styles_match']}% - Personality-driven work approaches")
            report.append(f"🎉 OVERALL MATCH:       {top_match['overall_match']}%")
            report.append("")
            
            # Show all career matches for comparison
            report.append("🏆 ALL CAREER MATCHES RANKED:")
            report.append("-" * 50)
            for j, match in enumerate(result["matches"], 1):
                report.append(f"{j}. {match['job_name']}: {match['overall_match']}% overall")
                report.append(f"   Skills: {match['breakdown']['skills_match']}% | Values: {match['breakdown']['values_match']}% | Interests: {match['breakdown']['interests_match']}% | Work Styles: {match['breakdown']['work_styles_match']}%")
            report.append("")
            
            # MVP REQUIREMENT 3: Skills gaps and improvement courses
            if top_match["improvements"]:
                report.append("🎯 SKILLS GAPS & DEVELOPMENT RECOMMENDATIONS:")
                report.append("-" * 60)
                for imp in top_match["improvements"]:
                    report.append(f"• {imp['skill']} - {imp['gap_severity']} Priority Gap")
                    report.append(f"  📊 Current Level: {imp['current_level']}/5 → Target Level: {imp['required_level']}/5")
                    report.append(f"  💡 Strategy: {imp['improvement_tip']}")
                    report.append("")
                    
                    report.append("  🆓 FREE LEARNING RESOURCES:")
                    for course in imp['free_courses']:
                        report.append(f"    • {course['name']} - {course['provider']}")
                        report.append(f"      🔗 {course['url']}")
                    report.append("")
                    
                    report.append("  💰 PREMIUM COURSES:")
                    for course in imp['paid_courses']:
                        price = course.get('price', 'Contact for pricing')
                        report.append(f"    • {course['name']} - {course['provider']} ({price})")
                        report.append(f"      🔗 {course['url']}")
                    report.append("")
            else:
                report.append("🎯 SKILLS ANALYSIS:")
                report.append("-" * 30)
                report.append("✅ Excellent skill alignment! You meet or exceed all required skill levels for this role.")
                if top_match["strengths"]:
                    report.append("Key Strengths:")
                    for strength in top_match["strengths"]:
                        report.append(f"  • {strength}")
                report.append("")
            
            # MVP REQUIREMENT 4: Interview questions and answer templates
            interview_prep = top_match["interview_preparation"]
            report.append("🎤 INTERVIEW PREPARATION GUIDE:")
            report.append("-" * 50)
            
            # Skills-based questions
            if interview_prep["skills_questions"]:
                report.append("💪 SKILLS-BASED INTERVIEW QUESTIONS:")
                for q in interview_prep["skills_questions"]:
                    report.append(f"   • {q}")
                report.append(f"📝 Answer Template: {interview_prep['answer_templates']['skills']}")
                report.append("")
            
            # Values-based questions  
            if interview_prep["values_questions"]:
                report.append("❤️ VALUES-BASED INTERVIEW QUESTIONS:")
                for q in interview_prep["values_questions"]:
                    report.append(f"   • {q}")
                report.append(f"📝 Answer Template: {interview_prep['answer_templates']['values']}")
                report.append("")
            
            # Interests-based questions
            if interview_prep["interests_questions"]:
                report.append("🎯 INTERESTS-BASED INTERVIEW QUESTIONS:")
                for q in interview_prep["interests_questions"]:
                    report.append(f"   • {q}")
                report.append(f"📝 Answer Template: {interview_prep['answer_templates']['interests']}")
                report.append("")
            
            # Work styles questions
            if interview_prep["work_styles_questions"]:
                report.append("⚙️ WORK STYLES INTERVIEW QUESTIONS:")
                for q in interview_prep["work_styles_questions"]:
                    report.append(f"   • {q}")
                report.append(f"📝 Answer Template: {interview_prep['answer_templates']['work_styles']}")
                report.append("")
            
            # Overall interview strategy
            report.append("🎯 OVERALL INTERVIEW STRATEGY:")
            report.append("-" * 40)
            report.append(f"Based on your {top_match['overall_match']}% match with {top_match['job_name']}, focus on:")
            
            # Identify strongest category
            strongest_category = max(breakdown, key=breakdown.get)
            category_names = {
                'skills_match': 'technical skills and competencies',
                'values_match': 'alignment with company values and culture',
                'interests_match': 'genuine passion and curiosity for the work',
                'work_styles_match': 'how your personality fits the role requirements'
            }
            
            report.append(f"✅ Emphasize your {category_names[strongest_category]} ({breakdown[strongest_category]}% match)")
            
            # Identify improvement areas
            weakest_category = min(breakdown, key=breakdown.get)
            if breakdown[weakest_category] < 80:
                report.append(f"⚠️  Address questions about {category_names[weakest_category]} proactively")
            
            report.append("")
            report.append("🔥 SUCCESS TIPS:")
            report.append("   • Use the STAR method (Situation, Task, Action, Result) for behavioral questions")
            report.append("   • Prepare 3-5 specific examples that demonstrate your top skills")
            report.append("   • Research the company's values and connect them to your motivations")
            report.append("   • Ask thoughtful questions that show your genuine interest in the role")
            report.append("")
            
            report.append("\n" + "="*80 + "\n")
        
        # Summary section
        report.append("📋 ANALYSIS SUMMARY:")
        report.append("-" * 30)
        for i, result in enumerate(results, 1):
            top_match = result["matches"][0]
            report.append(f"{result['profile'].name}: Best match is {top_match['job_name']} ({top_match['overall_match']}% compatibility)")
        
        return "\n".join(report)
    
def main():
    """Main function to run the career matching system"""
    print("🎯 Career Matching System with O*NET Metrics")
    print("=" * 50)
    
    # Initialize the matcher
    matcher = CareerMatcher()
    
    # Get CSV file path
    csv_path = input("Enter the path to your Concierge.csv file: ").strip()
    
    if not csv_path:
        csv_path = "Concierge.csv"  # Default filename
    
    try:
        # Process the CSV file
        print(f"\nProcessing {csv_path}...")
        results = matcher.process_csv_file(csv_path)
        
        if not results:
            print("No valid profiles found in the CSV file.")
            return
        
        print(f"Successfully analyzed {len(results)} profiles!")
        
        # Generate report
        report = matcher.generate_report(results)
        
        # Save report to file
        output_filename = f"career_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📊 Analysis complete! Report saved to: {output_filename}")
        
        # Show summary
        print("\n" + "=" * 50)
        print("QUICK SUMMARY:")
        print("=" * 50)
        
        for result in results:
            profile = result["profile"]
            top_match = result["top_match"]
            print(f"{profile.name}: {top_match['job_name']} ({top_match['overall_match']}% match)")
        
        # Option to save as JSON
        save_json = input("\nWould you like to save detailed results as JSON? (y/n): ").lower()
        if save_json == 'y':
            json_filename = f"career_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                json_result = {
                    "name": result["profile"].name,
                    "email": result["profile"].email,
                    "university": result["profile"].university,
                    "preferred_career": result["profile"].preferred_career,
                    "matches": result["matches"],
                    "analysis_date": result["analysis_date"]
                }
                json_results.append(json_result)
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            print(f"📄 JSON data saved to: {json_filename}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()