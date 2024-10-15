import React from 'react';
import { ScrollView, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { ThemedView } from '@/components/ThemedView';
import { ThemedText } from '@/components/ThemedText';
import { Ionicons } from '@expo/vector-icons';

const menuItems = [
  { title: 'Workshops', icon: 'construct-outline', screen: 'Workshops' },
  { title: 'Labs', icon: 'flask-outline', screen: 'Labs' },
  { title: 'Leaderboard', icon: 'trophy-outline', screen: 'Leaderboard' },
  { title: 'Sandbox', icon: 'code-slash-outline', screen: 'Sandbox' },
  { title: 'Newsletter', icon: 'mail-outline', screen: 'Newsletter' },
  { title: 'Recommendations', icon: 'bulb-outline', screen: 'Recommendations' },
  { title: 'Analytics', icon: 'bar-chart-outline', screen: 'Analytics' },
  { title: 'Exams', icon: 'document-text-outline', screen: 'Exams' },
  { title: 'Careers', icon: 'briefcase-outline', screen: 'Careers' },
  { title: 'Tutor', icon: 'person-outline', screen: 'Tutor' },
];

export default function MoreScreen() {
  const navigation = useNavigation();

  return (
    <ThemedView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <ThemedText style={styles.title}>More</ThemedText>
        {menuItems.map((item, index) => (
          <TouchableOpacity
            key={index}
            style={styles.menuItem}
            onPress={() => navigation.navigate(item.screen)}
          >
            <Ionicons name={item.icon} size={24} color="#10B981" style={styles.menuIcon} />
            <ThemedText style={styles.menuText}>{item.title}</ThemedText>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  menuIcon: {
    marginRight: 16,
  },
  menuText: {
    fontSize: 16,
  },
});