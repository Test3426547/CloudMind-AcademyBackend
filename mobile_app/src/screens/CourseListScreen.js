import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';

const CourseListScreen = ({ navigation }) => {
  const [courses, setCourses] = useState([]);

  useEffect(() => {
    // TODO: Fetch courses from the API
    const fetchCourses = async () => {
      // Simulated API call
      const mockCourses = [
        { id: '1', title: 'Introduction to Python' },
        { id: '2', title: 'Web Development Fundamentals' },
        { id: '3', title: 'Data Science Basics' },
      ];
      setCourses(mockCourses);
    };

    fetchCourses();
  }, []);

  const renderCourseItem = ({ item }) => (
    <TouchableOpacity
      style={styles.courseItem}
      onPress={() => navigation.navigate('CourseDetails', { courseId: item.id })}
    >
      <Text style={styles.courseTitle}>{item.title}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Available Courses</Text>
      <FlatList
        data={courses}
        renderItem={renderCourseItem}
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
  courseItem: {
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
  courseTitle: {
    fontSize: 18,
  },
});

export default CourseListScreen;
