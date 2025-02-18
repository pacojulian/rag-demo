'use client'

import * as React from 'react'
import { FileUp, MessageSquare, Send } from 'lucide-react'

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from '@/components/ui/sidebar'
import { Textarea } from "@/components/ui/textarea"

export default function DocumentChat() {
  const [activeTab, setActiveTab] = React.useState<'ingest' | 'chat'>('ingest')
  const [messages, setMessages] = React.useState<Array<{ role: 'user' | 'assistant', content: string }>>([])
  const [input, setInput] = React.useState('')

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    // Handle file upload logic here
    console.log('File uploaded:', event.target.files?.[0])
  }

  const handleSendMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { role: 'user', content: input }])
      setInput('')
      // Here you would typically send the message to your AI backend
      // and then add the response to the messages
    }
  }

  return (
    <SidebarProvider>
      <div className="flex h-screen">
        <Sidebar>
          <SidebarHeader>
            <h2 className="px-4 text-lg font-semibold">Document AI</h2>
          </SidebarHeader>
          <SidebarContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => setActiveTab('ingest')}
                  isActive={activeTab === 'ingest'}
                >
                  <FileUp className="mr-2 h-4 w-4" />
                  Ingest Document
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  onClick={() => setActiveTab('chat')}
                  isActive={activeTab === 'chat'}
                >
                  <MessageSquare className="mr-2 h-4 w-4" />
                  Chat
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarContent>
        </Sidebar>
        <div className="flex-1 p-4">
          {activeTab === 'ingest' ? (
            <div>
              <h2 className="text-2xl font-bold mb-4">Ingest Document</h2>
              <form className="space-y-4">
                <div>
                  <Label htmlFor="file-upload">Upload Document</Label>
                  <Input
                    id="file-upload"
                    type="file"
                    onChange={handleFileUpload}
                  />
                </div>
                <Button type="submit">Ingest Document</Button>
              </form>
            </div>
          ) : (
            <div className="flex flex-col h-full">
              <h2 className="text-2xl font-bold mb-4">Chat</h2>
              <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`p-2 rounded-lg ${
                      message.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'
                    } max-w-[80%]`}
                  >
                    {message.content}
                  </div>
                ))}
              </div>
              <div className="flex items-center space-x-2">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message here..."
                  className="flex-1"
                />
                <Button onClick={handleSendMessage}>
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </SidebarProvider>
  )
}
