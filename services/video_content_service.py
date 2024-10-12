import asyncio
from typing import List, Dict, Any
from fastapi import HTTPException
import logging
from services.llm_orchestrator import LLMOrchestrator, get_llm_orchestrator
from services.text_embedding_service import TextEmbeddingService, get_text_embedding_service
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class VideoContentService:
    def __init__(self, llm_orchestrator: LLMOrchestrator, text_embedding_service: TextEmbeddingService):
        self.llm_orchestrator = llm_orchestrator
        self.text_embedding_service = text_embedding_service
        self.video_database = {}  # Simulated database

    async def upload_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            video_id = f"video_{len(self.video_database) + 1}"
            self.video_database[video_id] = video_data
            
            # Perform AI-powered content analysis
            content_analysis = await self.analyze_video_content(video_data['title'], video_data['description'])
            self.video_database[video_id].update(content_analysis)
            
            return {"video_id": video_id, "message": "Video uploaded successfully", "analysis": content_analysis}
        except Exception as e:
            logger.error(f"Error uploading video: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while uploading the video")

    async def get_video(self, video_id: str) -> Dict[str, Any]:
        if video_id not in self.video_database:
            raise HTTPException(status_code=404, detail="Video not found")
        return self.video_database[video_id]

    async def analyze_video_content(self, title: str, description: str) -> Dict[str, Any]:
        try:
            prompt = f"Analyze the following video content and provide:\n1. A list of relevant tags\n2. The main topic\n3. The target audience\n4. Difficulty level (Beginner, Intermediate, Advanced)\n\nTitle: {title}\nDescription: {description}\n\nAnalysis:"
            
            analysis = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI video content analyzer."},
                {"role": "user", "content": prompt}
            ], "medium")
            
            # Parse the analysis
            lines = analysis.strip().split('\n')
            tags = lines[0].split(':')[1].strip().split(', ')
            topic = lines[1].split(':')[1].strip()
            audience = lines[2].split(':')[1].strip()
            difficulty = lines[3].split(':')[1].strip()
            
            return {
                "tags": tags,
                "main_topic": topic,
                "target_audience": audience,
                "difficulty_level": difficulty
            }
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            return {"error": "Failed to analyze video content"}

    async def recommend_videos(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        try:
            # In a real-world scenario, we would fetch user preferences and watch history
            user_preferences = ["programming", "machine learning", "data science"]
            
            # Generate embeddings for user preferences
            user_embedding = await self.text_embedding_service.get_embedding(" ".join(user_preferences))
            
            # Generate embeddings for all videos
            video_embeddings = []
            for video_id, video_data in self.video_database.items():
                video_content = f"{video_data['title']} {video_data['description']} {' '.join(video_data['tags'])}"
                video_embedding = await self.text_embedding_service.get_embedding(video_content)
                video_embeddings.append((video_id, video_embedding))
            
            # Calculate similarity scores
            similarity_scores = []
            for video_id, video_embedding in video_embeddings:
                similarity = np.dot(user_embedding, video_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(video_embedding))
                similarity_scores.append((video_id, similarity))
            
            # Sort by similarity and get top recommendations
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [{"video_id": video_id, "similarity": float(similarity)} 
                               for video_id, similarity in similarity_scores[:num_recommendations]]
            
            return recommendations
        except Exception as e:
            logger.error(f"Error recommending videos: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while recommending videos")

    async def generate_video_summary(self, video_id: str) -> str:
        try:
            video_data = await self.get_video(video_id)
            prompt = f"Generate a concise summary of the following video content:\n\nTitle: {video_data['title']}\nDescription: {video_data['description']}\nTags: {', '.join(video_data['tags'])}\nMain Topic: {video_data['main_topic']}\nTarget Audience: {video_data['target_audience']}\nDifficulty Level: {video_data['difficulty_level']}\n\nSummary:"
            
            summary = await self.llm_orchestrator.process_request([
                {"role": "system", "content": "You are an AI video content summarizer."},
                {"role": "user", "content": prompt}
            ], "medium")
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while generating the video summary")

    async def cluster_videos(self, num_clusters: int = 5) -> Dict[str, List[str]]:
        try:
            # Generate embeddings for all videos
            video_embeddings = []
            video_ids = []
            for video_id, video_data in self.video_database.items():
                video_content = f"{video_data['title']} {video_data['description']} {' '.join(video_data['tags'])}"
                video_embedding = await self.text_embedding_service.get_embedding(video_content)
                video_embeddings.append(video_embedding)
                video_ids.append(video_id)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(video_embeddings)
            
            # Group videos by cluster
            clusters = {i: [] for i in range(num_clusters)}
            for video_id, label in zip(video_ids, cluster_labels):
                clusters[label].append(video_id)
            
            return clusters
        except Exception as e:
            logger.error(f"Error clustering videos: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while clustering videos")

video_content_service = VideoContentService(get_llm_orchestrator(), get_text_embedding_service())

def get_video_content_service() -> VideoContentService:
    return video_content_service
