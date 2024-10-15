import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { getARVRContent, startARVRSession } from '../services/api';

const ARVRScreen = ({ navigation }) => {
  const [arvrContent, setARVRContent] = useState([]);

  useEffect(() => {
    fetchARVRContent();
  }, []);

  const fetchARVRContent = async () => {
    try {
      const content = await getARVRContent();
      setARVRContent(content);
    } catch (error) {
      console.error('Error fetching AR/VR content:', error);
    }
  };

  const handleStartSession = async (contentId) => {
    try {
      const session = await startARVRSession(contentId);
      // Navigate to AR/VR viewer or launch external AR/VR application
      console.log('Started AR/VR session:', session);
    } catch (error) {
      console.error('Error starting AR/VR session:', error);
    }
  };

  const renderARVRItem = ({ item }) => (
    <TouchableOpacity
      style={styles.arvrItem}
      onPress={() => handleStartSession(item.id)}
    >
      <Text style={styles.arvrTitle}>{item.title}</Text>
      <Text style={styles.arvrDescription}>{item.description}</Text>
      <Text style={styles.arvrInfo}>Type: {item.content_type}</Text>
      <Text style={styles.arvrInfo}>Complexity: {item.complexity_level}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AR/VR Learning Experiences</Text>
      <FlatList
        data={arvrContent}
        renderItem={renderARVRItem}
        keyExtractor={(item) => item.id}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5FCFF',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  arvrItem: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 5,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  arvrTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  arvrDescription: {
    marginTop: 5,
    color: '#666',
  },
  arvrInfo: {
    marginTop: 5,
    color: '#888',
  },
});

export default ARVRScreen;
