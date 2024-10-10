import axios from 'axios';

const API_URL = 'http://localhost:8000/api/v1'; // Replace with your actual API URL

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ... (keep existing functions)

export const getNotificationPreferences = async () => {
  try {
    const response = await api.get('/notifications/preferences');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const updateNotificationPreferences = async (preferences) => {
  try {
    const response = await api.post('/notifications/preferences', preferences);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const getUserNotifications = async () => {
  try {
    const response = await api.get('/notifications');
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const markNotificationAsRead = async (notificationId) => {
  try {
    const response = await api.post(`/notifications/mark-read/${notificationId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const deleteNotification = async (notificationId) => {
  try {
    const response = await api.delete(`/notifications/${notificationId}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export default api;
