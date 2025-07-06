import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# NLP and ML libraries
import spacy
from spacy.matcher import Matcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import PyPDF2
import docx2txt

@dataclass
class ResumeScore:
    filename: str
    score: float
    match_percentage: float
    key_skills_found: List[str]
    experience_years: Optional[int]
    education_level: str
    recommendation: str

class ResumeProcessor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        
    def _setup_patterns(self):
        """Setup patterns for extracting information from resumes"""
        # Email pattern
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # Phone pattern
        phone_pattern = [{"TEXT": {"REGEX": r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"}}]
        self.matcher.add("PHONE", [phone_pattern])
        
        # Experience patterns
        exp_patterns = [
            [{"LOWER": {"IN": ["experience", "work", "employment"]}},
             {"IS_DIGIT": True}, 
             {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}}],
            [{"IS_DIGIT": True}, 
             {"LOWER": {"IN": ["years", "year", "yrs", "yr"]}},
             {"LOWER": {"IN": ["experience", "work", "employment"]}}]
        ]
        for pattern in exp_patterns:
            self.matcher.add("EXPERIENCE", [pattern])

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or DOCX files"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_from_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file"""
        try:
            return docx2txt.process(str(file_path))
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_skills(self, text: str, skill_list: List[str]) -> List[str]:
        """Extract skills from text based on predefined skill list"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in skill_list:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience from text"""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "EXPERIENCE":
                span = doc[start:end]
                # Extract numbers from the span
                numbers = [token.text for token in span if token.is_digit]
                if numbers:
                    return int(numbers[0])
        
        # Fallback: regex search
        patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|work)',
            r'(?:experience|work).*?(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return int(matches[0])
        
        return None
    
    def extract_education_level(self, text: str) -> str:
        """Extract education level from text"""
        text_lower = text.lower()
        
        education_keywords = {
            'PhD': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'Masters': ['master', 'msc', 'm.sc', 'ma', 'm.a', 'mba', 'm.b.a'],
            'Bachelors': ['bachelor', 'bsc', 'b.sc', 'ba', 'b.a', 'be', 'b.e', 'btech', 'b.tech'],
            'Diploma': ['diploma', 'certificate'],
            'High School': ['high school', 'secondary', '12th', 'grade 12']
        }
        
        for level, keywords in education_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return 'Not specified'

class ResumeScreeningTool:
    def __init__(self, threshold: float = 0.6):
        self.processor = ResumeProcessor()
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def prepare_job_description(self, job_desc: str) -> Dict:
        """Process job description and extract requirements"""
        doc = self.processor.nlp(job_desc)
        
        # Extract skills from job description
        # This is a simplified approach - in practice, you'd want a more comprehensive skill database
        common_skills = [
            'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react', 'angular',
            'nodejs', 'mongodb', 'postgresql', 'aws', 'azure', 'docker', 'kubernetes',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'data analysis', 'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi',
            'project management', 'agile', 'scrum', 'git', 'github', 'jenkins',
            'communication', 'leadership', 'teamwork', 'problem solving'
        ]
        
        required_skills = self.processor.extract_skills(job_desc, common_skills)
        
        # Extract experience requirements
        experience_required = self.processor.extract_experience_years(job_desc)
        
        # Extract education requirements
        education_required = self.processor.extract_education_level(job_desc)
        
        return {
            'description': job_desc,
            'required_skills': required_skills,
            'experience_required': experience_required,
            'education_required': education_required
        }
    
    def score_resume(self, resume_text: str, job_requirements: Dict, filename: str) -> ResumeScore:
        """Score a resume against job requirements"""
        # Extract information from resume
        resume_skills = self.processor.extract_skills(resume_text, job_requirements['required_skills'])
        experience_years = self.processor.extract_experience_years(resume_text)
        education_level = self.processor.extract_education_level(resume_text)
        
        # Calculate skill match percentage
        if job_requirements['required_skills']:
            skill_match = len(resume_skills) / len(job_requirements['required_skills'])
        else:
            skill_match = 0
        
        # Calculate text similarity using TF-IDF
        job_desc = job_requirements['description']
        texts = [job_desc, resume_text]
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0
        
        # Calculate experience score
        exp_score = 1.0
        if job_requirements['experience_required'] and experience_years:
            if experience_years >= job_requirements['experience_required']:
                exp_score = 1.0
            else:
                exp_score = experience_years / job_requirements['experience_required']
        elif job_requirements['experience_required'] and not experience_years:
            exp_score = 0.3
        
        # Calculate final score (weighted combination)
        final_score = (
            0.4 * skill_match +
            0.3 * similarity +
            0.3 * exp_score
        )
        
        # Determine recommendation
        if final_score >= 0.8:
            recommendation = "Highly Recommended"
        elif final_score >= 0.6:
            recommendation = "Recommended"
        elif final_score >= 0.4:
            recommendation = "Consider"
        else:
            recommendation = "Not Recommended"
        
        return ResumeScore(
            filename=filename,
            score=final_score,
            match_percentage=skill_match * 100,
            key_skills_found=resume_skills,
            experience_years=experience_years,
            education_level=education_level,
            recommendation=recommendation
        )
    
    def screen_resumes(self, resume_folder: str, job_description: str) -> List[ResumeScore]:
        """Screen all resumes in a folder against job requirements"""
        resume_folder = Path(resume_folder)
        job_requirements = self.prepare_job_description(job_description)
        
        results = []
        supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        
        resume_files = []
        for fmt in supported_formats:
            resume_files.extend(resume_folder.glob(f'*{fmt}'))
        
        print(f"Found {len(resume_files)} resume files")
        
        for i, resume_file in enumerate(resume_files, 1):
            print(f"Processing {i}/{len(resume_files)}: {resume_file.name}")
            
            try:
                resume_text = self.processor.extract_text_from_file(resume_file)
                if resume_text.strip():
                    score = self.score_resume(resume_text, job_requirements, resume_file.name)
                    results.append(score)
                else:
                    print(f"Warning: No text extracted from {resume_file.name}")
            except Exception as e:
                print(f"Error processing {resume_file.name}: {e}")
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def save_results(self, results: List[ResumeScore], output_file: str):
        """Save screening results to a JSON file"""
        results_dict = []
        for result in results:
            results_dict.append({
                'filename': result.filename,
                'score': result.score,
                'match_percentage': result.match_percentage,
                'key_skills_found': result.key_skills_found,
                'experience_years': result.experience_years,
                'education_level': result.education_level,
                'recommendation': result.recommendation
            })
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_results(self, results: List[ResumeScore]):
        """Print screening results to console"""
        print("\n" + "="*80)
        print("RESUME SCREENING RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.filename}")
            print(f"   Score: {result.score:.2f}")
            print(f"   Skill Match: {result.match_percentage:.1f}%")
            print(f"   Experience: {result.experience_years or 'Not specified'} years")
            print(f"   Education: {result.education_level}")
            print(f"   Skills Found: {', '.join(result.key_skills_found) if result.key_skills_found else 'None'}")
            print(f"   Recommendation: {result.recommendation}")
        
        print("\n" + "="*80)
        print(f"Total resumes processed: {len(results)}")
        if results:
            avg_score = sum(r.score for r in results) / len(results)
            print(f"Average score: {avg_score:.2f}")
            recommended = len([r for r in results if r.score >= self.threshold])
            print(f"Recommended candidates: {recommended}")

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Resume Screening Tool')
    parser.add_argument('--resumes', '-r', required=True, help='Path to folder containing resumes')
    parser.add_argument('--job-desc', '-j', required=True, help='Job description text file or direct text')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, help='Minimum score threshold (default: 0.6)')
    parser.add_argument('--output', '-o', help='Output file to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Read job description
    if os.path.isfile(args.job_desc):
        with open(args.job_desc, 'r') as f:
            job_description = f.read()
    else:
        job_description = args.job_desc
    
    # Initialize screening tool
    screening_tool = ResumeScreeningTool(threshold=args.threshold)
    
    # Screen resumes
    print("Starting resume screening process...")
    print(f"Threshold: {args.threshold}")
    print(f"Resume folder: {args.resumes}")
    
    results = screening_tool.screen_resumes(args.resumes, job_description)
    
    # Display results
    screening_tool.print_results(results)
    
    # Save results if output file specified
    if args.output:
        screening_tool.save_results(results, args.output)
    
    print("\nResume screening completed!")

if __name__ == "__main__":
    main()

# Example usage:
# python resume_screening.py --resumes ./resumes --job-desc "job_description.txt" --threshold 0.7 --output results.json