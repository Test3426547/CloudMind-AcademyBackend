import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, TouchableOpacity } from 'react-native';
import { getLeaderboard, shareContent } from '../services/api';

const LeaderboardScreen = () => {
  const [leaderboard, setLeaderboard] = useState([]);
  const [period, setPeriod] = useState('all_time');

  useEffect(() => {
    fetchLeaderboard();
  }, [period]);

  const fetchLeaderboard = async () => {
    try {
      const data = await getLeaderboard(period);
      setLeaderboard(data.leaderboard);
    } catch (error) {
      console.error('Error fetching leaderboard:', error);
    }
  };

  const handleShare = async (userId, rank) => {
    try {
      await shareContent('leaderboard', `${userId}_${rank}`, 'twitter');
      alert('Leaderboard position shared successfully!');
    } catch (error) {
      console.error('Error sharing leaderboard position:', error);
    }
  };

  const renderLeaderboardItem = ({ item, index }) => (
    <View style={styles.leaderboardItem}>
      <Text style={styles.rank}>{index + 1}</Text>
      <Text style={styles.username}>{item.user_id}</Text>
      <Text style={styles.points}>{item.points} pts</Text>
      <TouchableOpacity onPress={() => handleShare(item.user_id, index + 1)}>
        <Text style={styles.shareButton}>Share</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Leaderboard</Text>
      <View style={styles.periodSelector}>
        <TouchableOpacity onPress={() => setPeriod('all_time')} style={[styles.periodButton, period === 'all_time' && styles.activePeriod]}>
          <Text>All Time</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setPeriod('monthly')} style={[styles.periodButton, period === 'monthly' && styles.activePeriod]}>
          <Text>Monthly</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setPeriod('weekly')} style={[styles.periodButton, period === 'weekly' && styles.activePeriod]}>
          <Text>Weekly</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => setPeriod('daily')} style={[styles.periodButton, period === 'daily' && styles.activePeriod]}>
          <Text>Daily</Text>
        </TouchableOpacity>
      </View>
      <FlatList
        data={leaderboard}
        renderItem={renderLeaderboardItem}
        keyExtractor={(item, index) => `${item.user_id}_${index}`}
      />
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
  periodSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  periodButton: {
    padding: 10,
    borderRadius: 5,
    backgroundColor: '#e0e0e0',
  },
  activePeriod: {
    backgroundColor: '#007AFF',
  },
  leaderboardItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  rank: {
    fontSize: 18,
    fontWeight: 'bold',
    width: 30,
  },
  username: {
    fontSize: 16,
    flex: 1,
  },
  points: {
    fontSize: 16,
    fontWeight: 'bold',
    marginRight: 10,
  },
  shareButton: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
});

export default LeaderboardScreen;
