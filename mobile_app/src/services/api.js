import axios from 'axios';
import { createClient } from '@supabase/supabase-js';

const API_URL = 'http://localhost:8000/api/v1'; // Replace with your actual API URL

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

export const signInWithOAuth = async (provider) => {
  try {
    const { data, error } = await supabase.auth.signInWithOAuth({ provider });
    if (error) throw error;
    return data;
  } catch (error) {
    throw error;
  }
};

export const signOut = async () => {
  try {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
  } catch (error) {
    throw error;
  }
};

export const getSession = async () => {
  try {
    const { data, error } = await supabase.auth.getSession();
    if (error) throw error;
    return data.session;
  } catch (error) {
    throw error;
  }
};

// Update the existing API calls to include the authentication token
const getAuthHeader = async () => {
  const session = await getSession();
  return session ? { Authorization: `Bearer ${session.access_token}` } : {};
};

export const getProductivityDashboard = async () => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.get('/time-tracking/dashboard', {
      headers: authHeader,
      params: {
        start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        end_date: new Date().toISOString(),
      },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const startTimer = async (taskId, description) => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.post('/time-tracking/start', { task_id: taskId, description }, { headers: authHeader });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const stopTimer = async (entryId) => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.post(`/time-tracking/${entryId}/stop`, {}, { headers: authHeader });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const chatWithTutor = async (message) => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.post('/ai-tutor/chat', { message }, { headers: authHeader });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getLeaderboard = async (period) => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.get('/leaderboard', {
      headers: authHeader,
      params: { period },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const shareContent = async (contentType, contentId, platform) => {
  try {
    const authHeader = await getAuthHeader();
    const response = await api.post('/share', {
      content_type: contentType,
      content_id: contentId,
      platform,
    }, { headers: authHeader });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
