import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity } from 'react-native';
import Voice from '@react-native-voice/voice';
import { chatWithTutor } from '../services/api';

const AITutorScreen = () => {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    Voice.onSpeechResults = onSpeechResults;
    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  const onSpeechResults = (e) => {
    setInput(e.value[0]);
  };

  const startListening = async () => {
    try {
      await Voice.start('en-US');
      setIsListening(true);
    } catch (e) {
      console.error(e);
    }
  };

  const stopListening = async () => {
    try {
      await Voice.stop();
      setIsListening(false);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSubmit = async () => {
    try {
      const result = await chatWithTutor(input);
      setResponse(result.response);
      setInput('');
    } catch (error) {
      console.error('Error chatting with AI Tutor:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Tutor</Text>
      <TextInput
        style={styles.input}
        value={input}
        onChangeText={setInput}
        placeholder="Ask your question here..."
        multiline
      />
      <TouchableOpacity
        style={styles.button}
        onPress={isListening ? stopListening : startListening}
      >
        <Text style={styles.buttonText}>
          {isListening ? 'Stop Voice Input' : 'Start Voice Input'}
        </Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.button} onPress={handleSubmit}>
        <Text style={styles.buttonText}>Submit</Text>
      </TouchableOpacity>
      <Text style={styles.responseTitle}>AI Tutor Response:</Text>
      <Text style={styles.response}>{response}</Text>
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
  input: {
    height: 100,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 10,
    padding: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 5,
    marginBottom: 10,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
  },
  responseTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 20,
  },
  response: {
    marginTop: 10,
    fontSize: 16,
  },
});

export default AITutorScreen;
