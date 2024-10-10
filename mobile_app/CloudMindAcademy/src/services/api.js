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

export const getLeaderboard = async () => {
  try {
    const response = await api.get('/gamification/leaderboard');
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

export default api;
