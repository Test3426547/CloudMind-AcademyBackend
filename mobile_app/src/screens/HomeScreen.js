import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { getProductivityDashboard } from '../services/api';

const HomeScreen = () => {
  const [dashboardData, setDashboardData] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const data = await getProductivityDashboard();
      setDashboardData(data);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };

  if (!dashboardData) {
    return <Text>Loading...</Text>;
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Welcome to CloudMind Academy</Text>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Productivity Overview</Text>
        <Text>Total Time: {(dashboardData.analytics.total_time / 3600).toFixed(2)} hours</Text>
        <Text>Productivity Score: {dashboardData.analytics.productivity_score.toFixed(2)}/100</Text>
        <Text>Productivity Trend: {dashboardData.analytics.productivity_trend}</Text>
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Recent Activities</Text>
        {dashboardData.recent_entries.map((entry, index) => (
          <Text key={index}>{entry.description} - {(entry.duration / 60).toFixed(2)} minutes</Text>
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
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
});

export default HomeScreen;
