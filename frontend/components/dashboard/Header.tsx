import { useState } from 'react'
import { Bell, Search, Menu } from 'lucide-react'
import { Input } from "@/components/ui/dashboard/input"
import { Button } from "@/components/ui/dashboard/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/dashboard/avatar"

interface HeaderProps {
  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void
}

export default function Header({ sidebarOpen, setSidebarOpen }: HeaderProps) {
  const [searchFocused, setSearchFocused] = useState(false)

  return (
    <header className="flex items-center justify-between px-6 py-4 bg-gray-800 border-b border-gray-700">
      <div className="flex items-center flex-1">
        <button
          className="text-gray-400 focus:outline-none lg:hidden mr-4"
          onClick={() => setSidebarOpen(true)}
        >
          <Menu className="h-6 w-6" />
        </button>

        <h3 className="text-gray-100 text-3xl font-semibold mr-4">Dashboard</h3>

        <div className="relative flex-1 max-w-2xl mx-auto">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center">
            <Search className={`h-5 w-5 ${searchFocused ? 'text-emerald-500' : 'text-gray-400'}`} />
          </span>

          <Input
            type="text"
            className="form-input w-full rounded-md pl-10 pr-4 bg-gray-700 border-gray-600 text-gray-200 focus:border-emerald-500"
            placeholder="Search"
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
          />
        </div>
      </div>

      <div className="flex items-center">
        <Button variant="ghost" size="icon" className="mr-4 relative text-gray-400 hover:text-gray-200">
          <Bell className="h-6 w-6" />
          <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-red-500"></span>
        </Button>

        <div className="relative">
          <Button variant="ghost" size="icon">
            <Avatar>
              <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
              <AvatarFallback>CN</AvatarFallback>
            </Avatar>
          </Button>
        </div>
      </div>
    </header>
  )
}