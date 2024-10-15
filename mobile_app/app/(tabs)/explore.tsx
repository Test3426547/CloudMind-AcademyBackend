import React from "react";
import { StyleSheet, ScrollView, View, TouchableOpacity } from "react-native";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";

import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { Card } from "@/components/Card";
import { SearchBar } from "@/components/SearchBar";

const categories = [
  { name: "Full Stack", icon: "layers-outline" },
  { name: "Frontend", icon: "desktop-outline" },
  { name: "Backend", icon: "server-outline" },
  { name: "Mobile", icon: "phone-portrait-outline" },
  { name: "DevOps", icon: "git-branch-outline" },
  { name: "AI/ML", icon: "brain-outline" },
];

const featuredCourses = [
  {
    id: "1",
    title: "Full Stack JIRA Clone",
    chapters: 41,
    image: require("@/assets/images/jira-clone.jpg"),
  },
  {
    id: "2",
    title: "Full Stack Slack Clone",
    chapters: 42,
    image: require("@/assets/images/slack-clone.jpg"),
  },
  {
    id: "3",
    title: "Full Stack Canva Clone",
    chapters: 52,
    image: require("@/assets/images/canva-clone.jpg"),
  },
];

export default function ExploreScreen() {
  const navigation = useNavigation();

  const navigateToCourse = (courseId) => {
    // Navigate to course details screen
    navigation.navigate("CourseDetails", { courseId });
  };

  return (
    <ThemedView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <ThemedText style={styles.title}>Explore</ThemedText>
        <SearchBar placeholder="Search for courses..." />

        <ThemedText style={styles.sectionTitle}>Categories</ThemedText>
        <View style={styles.categoriesContainer}>
          {categories.map((category, index) => (
            <TouchableOpacity key={index} style={styles.categoryItem}>
              <Ionicons name={category.icon} size={24} color="#10B981" />
              <ThemedText style={styles.categoryText}>
                {category.name}
              </ThemedText>
            </TouchableOpacity>
          ))}
        </View>

        <ThemedText style={styles.sectionTitle}>Featured Courses</ThemedText>
        {featuredCourses.map((course) => (
          <Card
            key={course.id}
            style={styles.courseCard}
            onPress={() => navigateToCourse(course.id)}
          >
            <Card.Cover source={course.image} />
            <Card.Content>
              <ThemedText style={styles.courseTitle}>{course.title}</ThemedText>
              <ThemedText style={styles.courseChapters}>
                {course.chapters} Chapters
              </ThemedText>
            </Card.Content>
          </Card>
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
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    marginTop: 24,
    marginBottom: 16,
  },
  categoriesContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
  },
  categoryItem: {
    width: "30%",
    alignItems: "center",
    marginBottom: 16,
  },
  categoryText: {
    marginTop: 8,
    textAlign: "center",
  },
  courseCard: {
    marginBottom: 16,
  },
  courseTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 4,
  },
  courseChapters: {
    fontSize: 14,
    color: "#6B7280",
  },
});
