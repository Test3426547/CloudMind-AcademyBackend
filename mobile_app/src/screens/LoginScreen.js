import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { signInWithOAuth } from '../services/api';

const LoginScreen = ({ navigation }) => {
  const handleOAuthLogin = async (provider) => {
    try {
      const { user, session } = await signInWithOAuth(provider);
      if (user && session) {
        navigation.navigate('Main');
      }
    } catch (error) {
      console.error('Error during OAuth login:', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to CloudMind Academy</Text>
      <TouchableOpacity
        style={styles.button}
        onPress={() => handleOAuthLogin('google')}
      >
        <Text style={styles.buttonText}>Sign in with Google</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => handleOAuthLogin('github')}
      >
        <Text style={styles.buttonText}>Sign in with GitHub</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#007AFF',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
    width: 200,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    textAlign: 'center',
  },
});

export default LoginScreen;
