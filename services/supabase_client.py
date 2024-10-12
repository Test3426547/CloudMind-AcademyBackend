import os
from supabase import create_client, Client
from typing import List, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from services.llm_orchestrator import get_llm_orchestrator
from services.text_embedding_service import get_text_embedding_service
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EnhancedSupabaseClient:
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
        self.llm_orchestrator = get_llm_orchestrator()
        self.text_embedding_service = get_text_embedding_service()
        self.scaler = StandardScaler()

    async def fetch_and_cluster_users(self, num_clusters: int = 3) -> Dict[str, Any]:
        try:
            # Fetch user data from Supabase
            response = self.supabase.table("users").select("id, name, age, interests").execute()
            users = response.data

            # Prepare data for clustering
            features = []
            for user in users:
                user_vector = [user['age']]
                interests_embedding = await self.text_embedding_service.get_embedding(user['interests'])
                user_vector.extend(interests_embedding)
                features.append(user_vector)

            # Normalize features
            features_normalized = self.scaler.fit_transform(features)

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_normalized)

            # Add cluster labels to user data
            for i, user in enumerate(users):
                user['cluster'] = int(cluster_labels[i])

            return {"users": users, "num_clusters": num_clusters}
        except Exception as e:
            logger.error(f"Error in fetch_and_cluster_users: {str(e)}")
            raise

    async def generate_user_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            # Fetch user data
            user_response = self.supabase.table("users").select("*").eq("id", user_id).execute()
            user = user_response.data[0] if user_response.data else None

            if not user:
                raise ValueError(f"User with id {user_id} not found")

            # Fetch all courses
            courses_response = self.supabase.table("courses").select("*").execute()
            courses = courses_response.data

            # Generate user embedding
            user_embedding = await self.text_embedding_service.get_embedding(user['interests'])

            # Generate course embeddings
            course_embeddings = []
            for course in courses:
                course_embedding = await self.text_embedding_service.get_embedding(f"{course['title']} {course['description']}")
                course_embeddings.append((course, course_embedding))

            # Calculate similarity scores
            similarity_scores = []
            for course, course_embedding in course_embeddings:
                similarity = cosine_similarity([user_embedding], [course_embedding])[0][0]
                similarity_scores.append((course, similarity))

            # Sort courses by similarity
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Generate recommendations using LLM
            top_courses = similarity_scores[:5]
            courses_text = "\n".join([f"{course['id']}: {course['title']} - {course['description']}" for course, _ in top_courses])

            prompt = f"""
            Given the following user interests and top 5 most similar courses, provide personalized recommendations and explanations.

            User interests: {user['interests']}

            Top 5 courses:
            {courses_text}

            Recommendations:
            """

            recommendations = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI course recommendation system."},
                {"role": "user", "content": prompt}
            ], "high")

            # Parse recommendations
            parsed_recommendations = []
            for line in recommendations.split("\n"):
                if line.strip():
                    course_id, explanation = line.split(":", 1)
                    parsed_recommendations.append({
                        "course_id": course_id.strip(),
                        "explanation": explanation.strip()
                    })

            return parsed_recommendations
        except Exception as e:
            logger.error(f"Error in generate_user_recommendations: {str(e)}")
            raise

    async def analyze_course_difficulty(self, course_id: str) -> Dict[str, Any]:
        try:
            # Fetch course data
            course_response = self.supabase.table("courses").select("*").eq("id", course_id).execute()
            course = course_response.data[0] if course_response.data else None

            if not course:
                raise ValueError(f"Course with id {course_id} not found")

            # Generate course embedding
            course_embedding = await self.text_embedding_service.get_embedding(f"{course['title']} {course['description']} {course.get('syllabus', '')}")

            # Analyze course content using LLM
            prompt = f"""
            Analyze the following course content and provide:
            1. An estimated difficulty level (Beginner, Intermediate, Advanced)
            2. Prerequisite knowledge or skills required
            3. Estimated time to complete the course (in hours)
            4. Key topics covered

            Course content:
            Title: {course['title']}
            Description: {course['description']}
            Syllabus: {course.get('syllabus', 'Not provided')}

            Analysis:
            """

            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI course content analyzer."},
                {"role": "user", "content": prompt}
            ], "high")

            # Parse analysis
            lines = analysis.split("\n")
            difficulty = lines[0].split(":")[1].strip()
            prerequisites = lines[1].split(":")[1].strip()
            estimated_time = int(lines[2].split(":")[1].strip().split()[0])  # Extract hours as integer
            key_topics = [topic.strip() for topic in lines[3].split(":")[1].strip().split(",")]

            return {
                "course_id": course_id,
                "difficulty": difficulty,
                "prerequisites": prerequisites,
                "estimated_time": estimated_time,
                "key_topics": key_topics,
                "embedding": course_embedding
            }
        except Exception as e:
            logger.error(f"Error in analyze_course_difficulty: {str(e)}")
            raise

    async def find_similar_courses(self, course_id: str, num_similar: int = 5) -> List[Dict[str, Any]]:
        try:
            # Fetch the target course
            target_course_response = self.supabase.table("courses").select("*").eq("id", course_id).execute()
            target_course = target_course_response.data[0] if target_course_response.data else None

            if not target_course:
                raise ValueError(f"Course with id {course_id} not found")

            # Generate target course embedding
            target_embedding = await self.text_embedding_service.get_embedding(f"{target_course['title']} {target_course['description']}")

            # Fetch all other courses
            all_courses_response = self.supabase.table("courses").select("*").neq("id", course_id).execute()
            all_courses = all_courses_response.data

            # Calculate similarity scores
            similarity_scores = []
            for course in all_courses:
                course_embedding = await self.text_embedding_service.get_embedding(f"{course['title']} {course['description']}")
                similarity = cosine_similarity([target_embedding], [course_embedding])[0][0]
                similarity_scores.append((course, similarity))

            # Sort courses by similarity and get top N
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_similar_courses = similarity_scores[:num_similar]

            # Prepare result
            result = []
            for course, similarity in top_similar_courses:
                result.append({
                    "course_id": course["id"],
                    "title": course["title"],
                    "similarity_score": similarity
                })

            return result
        except Exception as e:
            logger.error(f"Error in find_similar_courses: {str(e)}")
            raise

enhanced_supabase_client = EnhancedSupabaseClient()

def get_enhanced_supabase_client() -> EnhancedSupabaseClient:
    return enhanced_supabase_client
