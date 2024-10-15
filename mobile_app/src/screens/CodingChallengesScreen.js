import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity, TextInput } from 'react-native';
import { getCodingChallenges, submitCodingChallenge } from '../services/api';

const CodingChallengesScreen = ({ navigation }) => {
  const [challenges, setChallenges] = useState([]);
  const [selectedChallenge, setSelectedChallenge] = useState(null);
  const [userCode, setUserCode] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetchChallenges();
  }, []);

  const fetchChallenges = async () => {
    try {
      const data = await getCodingChallenges();
      setChallenges(data);
    } catch (error) {
      console.error('Error fetching coding challenges:', error);
    }
  };

  const handleChallengeSelect = (challenge) => {
    setSelectedChallenge(challenge);
    setUserCode(challenge.initial_code);
    setResult(null);
  };

  const handleSubmit = async () => {
    try {
      const submissionResult = await submitCodingChallenge(selectedChallenge.id, userCode);
      setResult(submissionResult);
    } catch (error) {
      console.error('Error submitting coding challenge:', error);
    }
  };

  const renderChallengeItem = ({ item }) => (
    <TouchableOpacity style={styles.challengeItem} onPress={() => handleChallengeSelect(item)}>
      <Text style={styles.challengeTitle}>{item.title}</Text>
      <Text style={styles.challengeDifficulty}>{item.difficulty}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      {!selectedChallenge ? (
        <>
          <Text style={styles.title}>Coding Challenges</Text>
          <FlatList
            data={challenges}
            renderItem={renderChallengeItem}
            keyExtractor={(item) => item.id}
          />
        </>
      ) : (
        <>
          <Text style={styles.title}>{selectedChallenge.title}</Text>
          <Text style={styles.description}>{selectedChallenge.description}</Text>
          <TextInput
            style={styles.codeInput}
            multiline
            value={userCode}
            onChangeText={setUserCode}
          />
          <TouchableOpacity style={styles.submitButton} onPress={handleSubmit}>
            <Text style={styles.submitButtonText}>Submit</Text>
          </TouchableOpacity>
          {result && (
            <View style={styles.resultContainer}>
              <Text style={styles.resultText}>
                {result.passed ? 'Challenge Passed!' : 'Challenge Failed'}
              </Text>
              <Text>Score: {result.score}</Text>
              <Text>Feedback: {result.feedback}</Text>
            </View>
          )}
        </>
      )}
    </View>
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
  challengeItem: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 5,
    marginBottom: 10,
  },
  challengeTitle: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  challengeDifficulty: {
    fontSize: 14,
    color: 'gray',
  },
  description: {
    marginBottom: 10,
  },
  codeInput: {
    height: 200,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    padding: 10,
    fontFamily: 'monospace',
  },
  submitButton: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 5,
    alignItems: 'center',
  },
  submitButtonText: {
    color: 'white',
    fontSize: 16,
  },
  resultContainer: {
    marginTop: 20,
    padding: 10,
    backgroundColor: '#E8E8E8',
    borderRadius: 5,
  },
  resultText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
});

export default CodingChallengesScreen;
