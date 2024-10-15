'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/dashboard/card'
import { Badge } from '@/components/ui/dashboard/badge'
import { Code } from 'lucide-react'

export function CodingChallengesComponent() {
  const challenges = [
    { name: 'AWS Lambda Function', difficulty: 'Easy', completed: true },
    { name: 'Google Cloud Storage', difficulty: 'Medium', completed: false },
    { name: 'Azure Functions', difficulty: 'Hard', completed: false },
  ]

  return (
    <Card className="bg-gray-800 border-gray-700">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-100">Coding Challenges</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {challenges.map((challenge, index) => (
            <div key={index} className="flex items-center justify-between">
              <div className="flex items-center">
                <Code className="h-5 w-5 text-emerald-500 mr-3" />
                <div>
                  <p className="text-sm font-medium text-gray-200">{challenge.name}</p>
                  <Badge variant={challenge.difficulty === 'Easy' ? 'secondary' : challenge.difficulty === 'Medium' ? 'default' : 'destructive'}>
                    {challenge.difficulty}
                  </Badge>
                </div>
              </div>
              <Badge variant={challenge.completed ? 'success' : 'outline'} className={challenge.completed ? 'bg-emerald-500 text-white' : 'text-gray-400 border-gray-600'}>
                {challenge.completed ? 'Completed' : 'Pending'}
              </Badge>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}