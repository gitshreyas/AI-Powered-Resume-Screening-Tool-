# setup.py - Run this first to install dependencies
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'spacy',
        'scikit-learn',
        'numpy',
        'pandas',
        'PyPDF2',
        'docx2txt',
        'pathlib'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    
    print("Setup complete!")

if __name__ == "__main__":
    install_packages()

# ========================================================================================
# sample_job_description.txt - Save this as a separate file
"""
Job Title: Senior Python Developer

We are looking for an experienced Python Developer to join our team. 

Requirements:
- 5+ years of experience in Python development
- Strong knowledge of machine learning and data analysis
- Experience with scikit-learn, pandas, numpy
- Familiarity with web frameworks like Django or Flask
- Experience with SQL databases (PostgreSQL, MySQL)
- Knowledge of cloud platforms (AWS, Azure)
- Strong problem-solving and communication skills
- Bachelor's degree in Computer Science or related field
- Experience with Git version control
- Understanding of agile development methodologies

Preferred Skills:
- Experience with TensorFlow or PyTorch
- Knowledge of Docker and Kubernetes
- Experience with REST API development
- Understanding of software testing principles
- Leadership experience is a plus

This is a full-time position offering competitive salary and benefits.
"""

# ========================================================================================
# create_sample_data.py - Script to create sample resume data
import os
from pathlib import Path

def create_sample_resumes():
    """Create sample resume files for testing"""
    
    # Create resumes directory
    resumes_dir = Path("sample_resumes")
    resumes_dir.mkdir(exist_ok=True)
    
    sample_resumes = {
        "john_doe.txt": """
John Doe
Senior Python Developer
Email: john.doe@email.com
Phone: +1-555-123-4567

EXPERIENCE
Senior Python Developer | TechCorp | 2019 - Present (6 years)
- Developed machine learning models using scikit-learn and pandas
- Built REST APIs using Django and Flask
- Worked with PostgreSQL and MySQL databases
- Deployed applications on AWS cloud platform
- Led team of 3 junior developers

Python Developer | StartupXYZ | 2017 - 2019 (2 years)
- Developed data analysis pipelines using pandas and numpy
- Created web applications using Django
- Implemented automated testing using pytest
- Used Git for version control

EDUCATION
Bachelor of Science in Computer Science
University of Technology, 2017

SKILLS
- Python, Django, Flask, scikit-learn, pandas, numpy
- PostgreSQL, MySQL, SQLite
- AWS, Docker, Git
- Machine Learning, Data Analysis
- REST API Development, Agile Methodologies
        """,
        
        "jane_smith.txt": """
Jane Smith
Data Scientist
Email: jane.smith@email.com
Phone: +1-555-987-6543

EXPERIENCE
Data Scientist | DataCorp | 2020 - Present (4 years)
- Built predictive models using scikit-learn and TensorFlow
- Performed data analysis using pandas, numpy, and matplotlib
- Developed machine learning pipelines for production
- Created data visualizations using Tableau and Power BI
- Collaborated with cross-functional teams using Agile methodologies

Junior Data Analyst | Analytics Inc | 2018 - 2020 (2 years)
- Analyzed large datasets using Python and SQL
- Created reports and dashboards for business stakeholders
- Worked with MySQL and PostgreSQL databases

EDUCATION
Master of Science in Data Science
Data University, 2018

Bachelor of Science in Mathematics
Math College, 2016

SKILLS
- Python, R, SQL, scikit-learn, TensorFlow, pandas, numpy
- MySQL, PostgreSQL, MongoDB
- Tableau, Power BI, matplotlib, seaborn
- Machine Learning, Deep Learning, Statistical Analysis
- Git, Jupyter, Docker
        """,
        
        "bob_johnson.txt": """
Bob Johnson
Junior Developer
Email: bob.johnson@email.com
Phone: +1-555-456-7890

EXPERIENCE
Junior Python Developer | WebDev Co | 2022 - Present (2 years)
- Developed web applications using Django
- Basic experience with Python and JavaScript
- Worked with SQLite databases
- Used Git for version control

Intern | Tech Startup | 2021 - 2022 (1 year)
- Assisted with basic web development tasks
- Learned HTML, CSS, and basic Python

EDUCATION
Bachelor of Science in Computer Science
Code University, 2021

SKILLS
- Python, Django, JavaScript, HTML, CSS
- SQLite, basic SQL knowledge
- Git, GitHub
- Basic web development
        """,
        
        "alice_wong.txt": """
Alice Wong
Machine Learning Engineer
Email: alice.wong@email.com
Phone: +1-555-111-2222

EXPERIENCE
Machine Learning Engineer | AI Solutions | 2018 - Present (7 years)
- Designed and implemented ML models using scikit-learn, TensorFlow, and PyTorch
- Built end-to-end ML pipelines using Python, pandas, and numpy
- Deployed models on AWS and Azure cloud platforms
- Developed REST APIs using Flask and FastAPI
- Led ML projects using Agile methodologies
- Mentored junior developers and data scientists

Data Scientist | BigData Corp | 2015 - 2018 (3 years)
- Performed statistical analysis and data mining
- Built predictive models for business intelligence
- Worked with large datasets using pandas and SQL
- Created data visualizations and reports

EDUCATION
PhD in Machine Learning
AI University, 2015

Master of Science in Computer Science
Tech Institute, 2012

SKILLS
- Python, R, scikit-learn, TensorFlow, PyTorch, pandas, numpy
- PostgreSQL, MySQL, MongoDB, Redis
- AWS, Azure, Docker, Kubernetes
- Flask, FastAPI, Django, REST APIs
- Machine Learning, Deep Learning, NLP, Computer Vision
- Git, Jenkins, Agile, Scrum, Leadership
        """,
        
        "mike_davis.txt": """
Mike Davis
Web Developer
Email: mike.davis@email.com
Phone: +1-555-333-4444

EXPERIENCE
Full Stack Developer | WebSolutions | 2020 - Present (4 years)
- Developed web applications using JavaScript, React, and Node.js
- Some experience with Python for automation tasks
- Worked with MongoDB and PostgreSQL
- Used Git for version control

Frontend Developer | DesignCorp | 2018 - 2020 (2 years)
- Built user interfaces using HTML, CSS, JavaScript, and React
- Collaborated with UX designers and backend developers

EDUCATION
Bachelor of Arts in Web Design
Design College, 2018

SKILLS
- JavaScript, React, Node.js, HTML, CSS
- Some Python experience
- MongoDB, PostgreSQL
- Git, GitHub, Agile
- Web Development, UI/UX Design
        """
    }
    
    # Write sample resumes
    for filename, content in sample_resumes.items():
        with open(resumes_dir / filename, 'w') as f:
            f.write(content.strip())
    
    print(f"Created {len(sample_resumes)} sample resumes in {resumes_dir}")

if __name__ == "__main__":
    create_sample_resumes()

# ========================================================================================
# run_example.py - Example of how to use the tool
import subprocess
import sys
from pathlib import Path

def run_example():
    """Run an example of the resume screening tool"""
    
    # Create sample data first
    print("Creating sample data...")
    create_sample_resumes()
    
    # Create job description file
    job_desc = """
Job Title: Senior Python Developer

We are looking for an experienced Python Developer to join our team. 

Requirements:
- 5+ years of experience in Python development
- Strong knowledge of machine learning and data analysis
- Experience with scikit-learn, pandas, numpy
- Familiarity with web frameworks like Django or Flask
- Experience with SQL databases (PostgreSQL, MySQL)
- Knowledge of cloud platforms (AWS, Azure)
- Strong problem-solving and communication skills
- Bachelor's degree in Computer Science or related field
- Experience with Git version control
- Understanding of agile development methodologies

Preferred Skills:
- Experience with TensorFlow or PyTorch
- Knowledge of Docker and Kubernetes
- Experience with REST API development
- Understanding of software testing principles
- Leadership experience is a plus
    """
    
    with open("job_description.txt", "w") as f:
        f.write(job_desc)
    
    print("Running resume screening tool...")
    
    # Run the screening tool
    cmd = [
        sys.executable, 
        "resume_screening.py",
        "--resumes", "sample_resumes",
        "--job-desc", "job_description.txt",
        "--threshold", "0.6",
        "--output", "screening_results.json"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_example()