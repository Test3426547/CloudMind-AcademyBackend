import { useState } from 'react'
import Head from 'next/head'
import { Sidebar } from '@/components/dashboard/Sidebar'
import Header from '@/components/dashboard/Header'
import { ScrollArea } from "@/components/ui/dashboard/scroll-area"
import { OverviewComponent } from '@/components/dashboard/Overview'
import { RecentCoursesComponent } from '@/components/dashboard/RecentCourses'
import { UpcomingExamsComponent } from '@/components/dashboard/UpcomingExams'
import { LearningProgressComponent } from '@/components/dashboard/LearningProgress'
import { CodingChallengesComponent } from '@/components/dashboard/CodingChallenges'

export default function Dashboard() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <>
      <Head>
        <title>Dashboard | CloudMind Academy</title>
        <meta name="description" content="CloudMind Academy Dashboard" />
        <link rel="icon" href="/favicon.ico" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
      </Head>

      <div className="flex h-screen bg-gray-900 font-inter">
        <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

        <div className="flex-1 flex flex-col overflow-hidden">
          <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

          <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-900">
            <div className="container mx-auto px-6 py-8">

              <ScrollArea className="mt-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  <OverviewComponent />
                  <RecentCoursesComponent />
                  <UpcomingExamsComponent />
                  <LearningProgressComponent />
                  <CodingChallengesComponent />
                </div>
              </ScrollArea>
            </div>
          </main>
        </div>
      </div>
    </>
  )
}