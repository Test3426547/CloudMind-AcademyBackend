'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/dashboard/card'
import { CalendarIcon } from 'lucide-react'

export function UpcomingExamsComponent() {
  const exams = [
    { name: 'AWS Certified Solutions Architect', date: '2024-03-15' },
    { name: 'Google Cloud Professional Cloud Architect', date: '2024-04-02' },
    { name: 'Microsoft Azure Administrator', date: '2024-04-20' },
  ]

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-100">Upcoming Exams</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {exams.map((exam, index) => (
            <div key={index} className="flex items-center">
              <CalendarIcon className="h-5 w-5 text-emerald-500 mr-3" />
              <div>
                <p className="text-sm font-medium text-gray-200">{exam.name}</p>
                <p className="text-xs text-gray-400">{exam.date}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}