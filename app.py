from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://swastikmoolya.42web.io"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KE"),  # your real key
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"  # supported model
)

# Your resume JSON
resume_json = {
    "name": "Swastik Bhoja Moolya",
    "title": "DevOps Engineer | Cloud Infrastructure | Automation | CI/CD Pipelines",
    "contact": {
        "location": "Ahmedabad, Gujarat",
        "phone": "+91 9327973365",
        "email": "moolyaswastik48@gmail.com",
        "website": "http://killmonger.fwh.is/",
        "linkedin": "https://www.linkedin.com/in/swastik-python-dev",
        "github": "https://github.com/Killmongers"
    },
    "summary": "B.Tech Computer Engineering graduate with hands-on experience in Python backend development, AWS, Docker, Nginx, and automation scripting. Skilled in Generative AI (LangChain, LLM APIs, vector search) and cloud deployment. Proven ability to deliver scalable backend systems and AI-driven solutions.",
    "skills": {
        "programming": ["Python", "JavaScript", "Bash", "HTML/CSS"],
        "web_backend_frameworks": ["Flask", "FastAPI", "REST APIs"],
        "devops_infra_tools": ["Docker", "Git", "GitHub Actions", "Jenkins", "PM2", "Nginx", "Systemctl"],
        "ai_ml_genai": ["LangChain", "HuggingFace Transformers", "Prompt Engineering", "Retrieval-Augmented Generation (RAG)", "Ollama", "FAISS Vector Store", "LLM"],
        "cloud_deployment": ["AWS EC2/Lightsail", "Let's Encrypt", "CI/CD Pipelines", "Linux"],
        "other": ["Linux CLI", "Shell Scripting", "Twilio API", "OpenCV"]
    },
    "experience": [
        {
            "company": "FlyAnyTrip (Startup, On-site – Vadodara)",
            "role": "Python Developer Intern",
            "duration": "Dec 2024 – June 2025",
            "achievements": [
                "Built & deployed WhatsApp chatbot for train tracking using Flask, Twilio, and IRCTC APIs",
                "Converted web app to Android app; published on Google Play Store",
                "Automated AWS Lightsail provisioning with Bash, reducing setup time by 80%",
                "Deployed Flask app with Nginx, PM2, and SSL, improving stability & response time"
            ]
        }
    ],
    "projects": [
        {
            "name": "WhatsApp Chatbot for Train & PNR Status",
            "duration": "Dec 2024 – Mar 2025",
            "type": "Internship Project",
            "description": "Python chatbot using APIs, AWS Lightsail deployment, WhatsApp integration, secured with Nginx + SSL"
        },
        {
            "name": "AI-Based Real-Time Fire Detection System",
            "duration": "Mar 2023 – Apr 2024",
            "type": "Final Year B.Tech Project",
            "description": "Real-time fire detection using OpenCV & CNNs, with email & sound alerts"
        },
        {
            "name": "E-Nursery Web Application (Dockerized PHP + MySQL Stack)",
            "duration": "Mar 2022",
            "type": "Diploma Final Project",
            "description": "PHP + MySQL plant sales platform with authentication & order management; Dockerized for deployment"
        }
    ],
    "education": [
        {
            "institution": "Parul University",
            "degree": "B.Tech - CSE",
            "score": "7.53 CGPA",
            "duration": "2022-2025"
        },
        {
            "institution": "LJ Polytechnic",
            "degree": "Diploma in Computer Engineering",
            "score": "8.28 CGPA",
            "duration": "2019-2022"
        }
    ],
    "certifications": [
        {
            "name": "Data Analytics with Python",
            "provider": "NPTEL",
            "year": "2024"
        },
        {
            "name": "AWS Cloud Quest: Cloud Practitioner",
            "provider": "Amazon Web Services Training and Certification",
            "year": "June 2025"
        },
        {
            "name": "Python And Flask Framework Complete Course",
            "provider": "Udemy",
            "year": "2024"
        },
        {
            "name": "Linux for DevOps Engineers and Developers",
            "provider": "Udemy",
            "year": "2025"
        }
    ],
    "languages": ["Hindi", "Gujarati", "English", "Kannada"]
}

user_question={}
MAX_QUESTIONS=3

prompt = ChatPromptTemplate.from_template("""
You are swastik or act like swastik and answers questions about swastik professional profile.

Instructions:
- Answer in 1-2 sentences only.
- Keep it concise and to the point.
- Do NOT mention “according to JSON” or anything similar.
- Provide useful, direct answers.

User question: {question}
Resume data: {resume}
""")
# Build chain
chain = prompt | llm
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host

    # initialize counter if new user
    if client_ip not in user_question:
        user_question[client_ip] = 0

    # check if user exceeded limit
    if user_question[client_ip] >= MAX_QUESTIONS:
        return {
            "response": f"❌ You have reached the maximum of {MAX_QUESTIONS} questions. No further answers can be provided."
        }

    # increment counter
    user_question[client_ip] += 1

    # generate response
    response = chain.invoke({
        "resume": json.dumps(resume_json, indent=2),
        "question": req.question
    })

    return {
        "response": response.content,
        "questions_left": MAX_QUESTIONS - user_question[client_ip]
    }

@app.get("/")
async def root():
    return {"message": "Resume Chatbot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
