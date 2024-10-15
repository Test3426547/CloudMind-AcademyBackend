import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Switch, ScrollView } from 'react-native';
import { getNotificationPreferences, updateNotificationPreferences } from '../services/api';

const NotificationSettingsScreen = () => {
  const [preferences, setPreferences] = useState({
    email_notifications: true,
    push_notifications: true,
    sms_notifications: false,
    notification_types: {
      course_updates: true,
      new_challenges: true,
      leaderboard_changes: true,
      achievement_unlocked: true,
      assignment_reminders: true
    }
  });

  useEffect(() => {
    fetchPreferences();
  }, []);

  const fetchPreferences = async () => {
    try {
      const userPreferences = await getNotificationPreferences();
      setPreferences(userPreferences);
    } catch (error) {
      console.error('Error fetching notification preferences:', error);
    }
  };

  const handleToggle = async (key, subKey = null) => {
    let updatedPreferences;
    if (subKey) {
      updatedPreferences = {
        ...preferences,
        notification_types: {
          ...preferences.notification_types,
          [subKey]: !preferences.notification_types[subKey]
        }
      };
    } else {
      updatedPreferences = { ...preferences, [key]: !preferences[key] };
    }
    setPreferences(updatedPreferences);
    try {
      await updateNotificationPreferences(updatedPreferences);
    } catch (error) {
      console.error('Error updating notification preferences:', error);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Notification Settings</Text>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notification Channels</Text>
        <View style={styles.preference}>
          <Text>Email Notifications</Text>
          <Switch
            value={preferences.email_notifications}
            onValueChange={() => handleToggle('email_notifications')}
          />
        </View>
        <View style={styles.preference}>
          <Text>Push Notifications</Text>
          <Switch
            value={preferences.push_notifications}
            onValueChange={() => handleToggle('push_notifications')}
          />
        </View>
        <View style={styles.preference}>
          <Text>SMS Notifications</Text>
          <Switch
            value={preferences.sms_notifications}
            onValueChange={() => handleToggle('sms_notifications')}
          />
        </View>
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notification Types</Text>
        {Object.entries(preferences.notification_types).map(([key, value]) => (
          <View key={key} style={styles.preference}>
            <Text>{key.replace('_', ' ').charAt(0).toUpperCase() + key.replace('_', ' ').slice(1)}</Text>
            <Switch
              value={value}
              onValueChange={() => handleToggle('notification_types', key)}
            />
          </View>
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#F5FCFF',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  preference: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
});

export default NotificationSettingsScreen;
