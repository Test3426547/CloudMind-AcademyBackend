'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/dashboard/card'
import { BookOpen, Award, Clock } from 'lucide-react'

export function OverviewComponent() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-200">Courses in Progress</CardTitle>
          <BookOpen className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-gray-100">4</div>
          <p className="text-xs text-gray-400">+2 from last month</p>
        </CardContent>
      </Card>
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-200">Certificates Earned</CardTitle>
          <Award className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-gray-100">7</div>
          <p className="text-xs text-gray-400">+1 this month</p>
        </CardContent>
      </Card>
      <Card className="bg-gray-800 border-gray-700">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium text-gray-200">Study Time</CardTitle>
          <Clock className="h-4 w-4 text-emerald-500" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-gray-100">32h</div>
          <p className="text-xs text-gray-400">+4h from last week</p>
        </CardContent>
      </Card>
    </div>
  )
}