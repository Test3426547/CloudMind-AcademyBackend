import React, { useEffect, useRef } from 'react'
import Link from 'next/link'
import { ScrollArea } from "@/components/ui/dashboard/scroll-area"
import { Button } from "@/components/ui/dashboard/button"
import { Home, BookOpen, Beaker, Code, Trophy, Box, Mail, Compass, BarChart2, FileText, Briefcase, HelpCircle, BarChart } from 'lucide-react'
import { gsap } from 'gsap'

const sidebarItems = [
  { icon: Home, label: 'Home', href: '/' },
  { icon: BookOpen, label: 'Browse', href: '/browse' },
  { icon: Beaker, label: 'Workshops', href: '/workshops' },
  { icon: Code, label: 'Labs', href: '/labs', locked: true },
  { icon: Trophy, label: 'Leaderboard', href: '/leaderboard' },
  { icon: Box, label: 'Sandbox', href: '/sandbox' },
  { icon: BarChart, label: 'Recommendations', href: '/recommendations', locked: true },
  { icon: Mail, label: 'Newsletter', href: '/newsletter' },
  { icon: Compass, label: 'Learning Paths', href: '/learning-paths' },
  { icon: BarChart2, label: 'Analytics', href: '/analytics', locked: true },
  { icon: FileText, label: 'Exams', href: '/exams' },
  { icon: HelpCircle, label: 'Tutor', href: '/tutor' },
  { icon: Briefcase, label: 'Careers', href: '/careers', locked: true },
]

export function Sidebar() {
  const sidebarRef = useRef(null)
  const navItemsRef = useRef([])

  useEffect(() => {
    const sidebar = sidebarRef.current
    const title = sidebar.querySelector('h1')
    const navItems = navItemsRef.current
    const upgradeButton = sidebar.querySelector('button')
    const upgradeText = sidebar.querySelector('p')

    gsap.set([title, ...navItems, upgradeButton, upgradeText], { opacity: 0, y: 20 })

    gsap.timeline()
      .to(title, { opacity: 1, y: 0, duration: 0.5 })
      .to(navItems, { opacity: 1, y: 0, stagger: 0.2, ease: "power3.out", duration: 0.5 }, "-=0.3")
      .to([upgradeButton, upgradeText], { opacity: 1, y: 0, stagger: 0.2, ease: "power3.out", duration: 0.5 }, "-=0.3")
  }, [])

  return (
    <div className="w-[314px] bg-gray-800 shadow-md flex flex-col h-screen" ref={sidebarRef}>
      <div className="p-4">
        <h1 className="text-2xl font-bold text-emerald-500">CloudMind Academy</h1>
      </div>
      <ScrollArea className="flex-grow overflow-y-auto" style={{ height: 'calc(100vh - 8rem - 50px)' }}>
        <nav className="mt-2 px-4">
          {sidebarItems.map((item, index) => (
            <Link
              key={index}
              href={item.href}
              className="flex items-center justify-between py-2 px-4 text-gray-300 hover:bg-gray-700 hover:text-emerald-500 mb-2 rounded-xl border-2 border-gray-700 font-bold transition-colors duration-200 ease-in-out hover:border-emerald-500"
              ref={el => navItemsRef.current[index] = el}
            >
              <div className="flex items-center">
                <item.icon className="mr-3 h-5 w-5" />
                {item.label}
              </div>
              {item.locked && (
                <span className="px-2 py-1 text-xs bg-gray-700 rounded-full">
                  Locked
                </span>
              )}
            </Link>
          ))}
        </nav>
      </ScrollArea>
      <div className="w-full p-4 bg-gray-800 border-t border-gray-700">
        <Button className="w-full bg-emerald-500 hover:bg-emerald-600 text-white">Upgrade to Pro</Button>
        <p className="mt-2 text-xs text-gray-400">Unlock all courses, get access to source code, and more.</p>
      </div>
    </div>
  )
}