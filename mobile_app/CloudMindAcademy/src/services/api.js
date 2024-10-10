import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1'; // Replace with your actual API URL

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const login = async (email, password) => {
  try {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchCourses = async () => {
  try {
    const response = await api.get('/courses');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const chatWithTutor = async (message) => {
  try {
    const response = await api.post('/tutor/chat', { message });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const generateQuiz = async (topic, numQuestions) => {
  try {
    const response = await api.post('/quiz/generate', { topic, num_questions: numQuestions });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getLeaderboard = async (period = 'all_time', limit = 10) => {
  try {
    const response = await api.get(`/gamification/leaderboard?period=${period}&limit=${limit}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getLearningPath = async (goal) => {
  try {
    const response = await api.post('/learning-path/generate', { goal });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const authenticateVoice = async (voiceSample) => {
  try {
    const response = await api.post('/auth/voice-biometrics', { voice_sample: voiceSample });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getARVRContent = async () => {
  try {
    const response = await api.get('/ar-vr/content');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const startARVRSession = async (contentId) => {
  try {
    const response = await api.post('/ar-vr/session', { content_id: contentId });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateARVRSession = async (sessionId, progress) => {
  try {
    const response = await api.put(`/ar-vr/session/${sessionId}`, { progress });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const shareContent = async (contentType, contentId, platform) => {
  try {
    const response = await api.post('/social/share', { content_type: contentType, content_id: contentId, platform });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const likeContent = async (contentType, contentId) => {
  try {
    const response = await api.post('/social/like', { content_type: contentType, content_id: contentId });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const commentOnContent = async (contentType, contentId, comment) => {
  try {
    const response = await api.post('/social/comment', { content_type: contentType, content_id: contentId, comment });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getCodingChallenges = async () => {
  try {
    const response = await api.get('/coding-challenges');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const submitCodingChallenge = async (challengeId, userCode) => {
  try {
    const response = await api.post('/coding-challenges/submit', { challenge_id: challengeId, user_code: userCode });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getChallengeLeaderboard = async (challengeId) => {
  try {
    const response = await api.get(`/coding-challenges/leaderboard/${challengeId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
