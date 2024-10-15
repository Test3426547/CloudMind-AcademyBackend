'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/dashboard/card'
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip } from 'recharts'

const data = [
  { name: 'Mon', hours: 3 },
  { name: 'Tue', hours: 2 },
  { name: 'Wed', hours: 4 },
  { name: 'Thu', hours: 3 },
  { name: 'Fri', hours: 5 },
  { name: 'Sat', hours: 2 },
  { name: 'Sun', hours: 1 },
]

export function LearningProgressComponent() {
  return (
    <Card className="col-span-2 bg-gray-800 border-gray-700">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-100">Weekly Learning Progress</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <XAxis dataKey="name" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip
              contentStyle={{ background: '#374151', border: '1px solid #4B5563', borderRadius: '4px' }}
              labelStyle={{ color: '#E5E7EB' }}
            />
            <Bar dataKey="hours" fill="#10B981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}