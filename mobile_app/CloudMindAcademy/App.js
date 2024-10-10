import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, StyleSheet } from 'react-native';
import NetInfo from '@react-native-community/netinfo';

import LoginScreen from './src/screens/LoginScreen';
import HomeScreen from './src/screens/HomeScreen';
import CourseListScreen from './src/screens/CourseListScreen';
import AITutorScreen from './src/screens/AITutorScreen';
import LeaderboardScreen from './src/screens/LeaderboardScreen';
import ARVRScreen from './src/screens/ARVRScreen';
import CodingChallengesScreen from './src/screens/CodingChallengesScreen';
import { offlineSyncService } from './src/services/offline_sync';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();

const OfflineIndicator = () => {
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setIsOffline(!state.isConnected);
    });

    return () => unsubscribe();
  }, []);

  if (!isOffline) return null;

  return (
    <View style={styles.offlineContainer}>
      <Text style={styles.offlineText}>You are offline</Text>
    </View>
  );
};

const MainTabs = () => (
  <>
    <OfflineIndicator />
    <Tab.Navigator>
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Courses" component={CourseListScreen} />
      <Tab.Screen name="AI Tutor" component={AITutorScreen} />
      <Tab.Screen name="Challenges" component={CodingChallengesScreen} />
      <Tab.Screen name="Leaderboard" component={LeaderboardScreen} />
      <Tab.Screen name="AR/VR" component={ARVRScreen} />
    </Tab.Navigator>
  </>
);

const App = () => {
  useEffect(() => {
    offlineSyncService.initializeOfflineData();

    const syncInterval = setInterval(() => {
      offlineSyncService.syncWithServer();
    }, 60000); // Try to sync every minute

    return () => clearInterval(syncInterval);
  }, []);

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Login">
        <Stack.Screen name="Login" component={LoginScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Main" component={MainTabs} options={{ headerShown: false }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  offlineContainer: {
    backgroundColor: '#b52424',
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'row',
    width: '100%',
    position: 'absolute',
    top: 0,
    zIndex: 1,
  },
  offlineText: {
    color: '#fff',
  },
});

export default App;
