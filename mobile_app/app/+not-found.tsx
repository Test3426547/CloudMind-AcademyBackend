import React from 'react';
import { Link, Stack } from 'expo-router';
import { StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Button } from '@/components/ui/button';

export default function NotFoundScreen() {
  return (
    <>
      <Stack.Screen options={{ title: 'Page Not Found' }} />
      <ThemedView style={styles.container}>
        <Ionicons name="alert-circle-outline" size={64} color="#10B981" style={styles.icon} />
        <ThemedText style={styles.title}>Oops! Page Not Found</ThemedText>
        <ThemedText style={styles.description}>
          The page you're looking for doesn't exist or has been moved.
        </ThemedText>
        <Link href="/" asChild>
          <Button style={styles.button}>
            <Ionicons name="home-outline" size={20} color="#FFFFFF" style={styles.buttonIcon} />
            <ThemedText style={styles.buttonText}>Go to Home</ThemedText>
          </Button>
        </Link>
      </ThemedView>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#F9FAFB',
  },
  icon: {
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#1F2937',
  },
  description: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 30,
    color: '#4B5563',
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#10B981',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  buttonIcon: {
    marginRight: 8,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});