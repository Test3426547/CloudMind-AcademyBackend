import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TextInput, Button } from 'react-native';
import { startTimer, stopTimer } from '../services/api';

const TimeTrackingScreen = () => {
  const [taskDescription, setTaskDescription] = useState('');
  const [currentTimer, setCurrentTimer] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    let interval;
    if (currentTimer) {
      interval = setInterval(() => {
        setElapsedTime((prevTime) => prevTime + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [currentTimer]);

  const handleStartTimer = async () => {
    try {
      const response = await startTimer('task_id', taskDescription);
      setCurrentTimer(response);
      setElapsedTime(0);
    } catch (error) {
      console.error('Error starting timer:', error);
    }
  };

  const handleStopTimer = async () => {
    if (currentTimer) {
      try {
        await stopTimer(currentTimer.id);
        setCurrentTimer(null);
        setElapsedTime(0);
        setTaskDescription('');
      } catch (error) {
        console.error('Error stopping timer:', error);
      }
    }
  };

  const formatTime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Time Tracking</Text>
      <TextInput
        style={styles.input}
        placeholder="Enter task description"
        value={taskDescription}
        onChangeText={setTaskDescription}
      />
      {currentTimer ? (
        <View>
          <Text style={styles.timerText}>{formatTime(elapsedTime)}</Text>
          <Button title="Stop Timer" onPress={handleStopTimer} />
        </View>
      ) : (
        <Button title="Start Timer" onPress={handleStartTimer} disabled={!taskDescription} />
      )}
    </View>
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
  input: {
    borderWidth: 1,
    borderColor: '#ccc',
    padding: 10,
    marginBottom: 20,
  },
  timerText: {
    fontSize: 36,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
});

export default TimeTrackingScreen;
