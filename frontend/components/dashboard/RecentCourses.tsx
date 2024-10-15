'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/dashboard/card'
import { Progress } from '@/components/ui/dashboard/progress'

export function RecentCoursesComponent() {
  const courses = [
    { name: 'AWS Certified Solutions Architect', progress: 75 },
    { name: 'Google Cloud Professional Cloud Architect', progress: 40 },
    { name: 'Microsoft Azure Administrator', progress: 60 },
  ]

  return (
    <Card className="col-span-2 bg-gray-800 border-gray-700">
      <CardHeader>
        <CardTitle className="text-lg font-semibold text-gray-100">Recent Courses</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {courses.map((course, index) => (
            <div key={index} className="flex items-center">
              <div className="w-full">
                <p className="text-sm font-medium text-gray-200">{course.name}</p>
                <Progress value={course.progress} className="h-2 mt-1 bg-gray-700">
                  <div
                    className="h-full bg-emerald-500 transition-all"
                    style={{ width: `${course.progress}%` }}
                  />
                </Progress>
              </div>
              <span className="ml-4 text-sm text-gray-400">{course.progress}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}