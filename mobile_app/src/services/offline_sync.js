import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import api from './api';

class OfflineSyncService {
  constructor() {
    this.syncQueue = [];
    this.isSyncing = false;
  }

  async initializeOfflineData() {
    try {
      const storedQueue = await AsyncStorage.getItem('syncQueue');
      if (storedQueue) {
        this.syncQueue = JSON.parse(storedQueue);
      }
    } catch (error) {
      console.error('Error initializing offline data:', error);
    }
  }

  async addToSyncQueue(data) {
    this.syncQueue.push(data);
    await this.saveSyncQueue();
  }

  async saveSyncQueue() {
    try {
      await AsyncStorage.setItem('syncQueue', JSON.stringify(this.syncQueue));
    } catch (error) {
      console.error('Error saving sync queue:', error);
    }
  }

  async syncWithServer() {
    if (this.isSyncing || this.syncQueue.length === 0) {
      return;
    }

    const isConnected = await NetInfo.fetch().then(state => state.isConnected);
    if (!isConnected) {
      return;
    }

    this.isSyncing = true;

    try {
      const syncData = {
        user_progress: {},
        quiz_responses: {}
      };

      this.syncQueue.forEach(item => {
        if (item.type === 'user_progress') {
          syncData.user_progress[item.course_id] = item.progress;
        } else if (item.type === 'quiz_response') {
          syncData.quiz_responses[item.quiz_id] = item.responses;
        }
      });

      const response = await api.post('/offline/sync', syncData);

      if (response.data.message === 'Sync successful') {
        this.syncQueue = [];
        await this.saveSyncQueue();
      }
    } catch (error) {
      console.error('Error syncing with server:', error);
    } finally {
      this.isSyncing = false;
    }
  }

  async getCachedCourseContent(courseId) {
    try {
      const cachedContent = await AsyncStorage.getItem(`course_${courseId}`);
      return cachedContent ? JSON.parse(cachedContent) : null;
    } catch (error) {
      console.error('Error getting cached course content:', error);
      return null;
    }
  }

  async cacheCourseContent(courseId, content) {
    try {
      await AsyncStorage.setItem(`course_${courseId}`, JSON.stringify(content));
    } catch (error) {
      console.error('Error caching course content:', error);
    }
  }
}

export const offlineSyncService = new OfflineSyncService();
