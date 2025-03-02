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
import axios from "axios";
import { AxiosError } from 'axios';

export default function DocumentChat() {
    const [activeTab, setActiveTab] = React.useState<'ingest' | 'chat'>('ingest')
    const [messages, setMessages] = React.useState<Array<{ role: 'user' | 'assistant', content: string }>>([])
    const [input, setInput] = React.useState('')
    const [loading, setLoading] = React.useState(false) // New loading state

    interface ISVGProps extends React.SVGProps<SVGSVGElement> {
        size?: number;
        className?: string;
    }

    const LoadingSpinner = ({
        size = 24,
        className,
        ...props
    }: ISVGProps) => {
        return (
            <svg
                xmlns="http://www.w3.org/2000/svg"
                width={size}
                height={size}
                {...props}
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={`animate-spin ${className}`}
            >
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
        );
    };

    const AxiosInstance = axios.create({
        baseURL: 'http://127.0.0.1:8000',
    });

    const chatApi = async (message: string) => {
        try {
            const requestBody = {
                query: message,
            };
            const response = await AxiosInstance.post('/chatbot/', requestBody);

            const parsedResponse = response.data;
            const innerResponse = JSON.parse(parsedResponse.response);

            return innerResponse.response;
        } catch (error: unknown) {
            if (error instanceof AxiosError && error.response && error.response.status === 400) {
                return 'Sorry, upload document first';
            }

            console.error('Unexpected error:', error);
            throw error;  // Optional: re-throw or handle differently
        }
    }

    const uplodFileApi = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);

        const response = await AxiosInstance.post('/upload_document/', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        return response.data
    }

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        uplodFileApi(event.target.files?.[0] as File).then((data) => {
            console.log(data);
            alert('File uploaded successfully')
        });
    }

    const handleSendMessage = () => {
        if (input.trim()) {
            setMessages(prevMessages => [...prevMessages, { role: 'user', content: input }]);
            setInput('');
            setLoading(true); // Set loading to true when sending the message

            chatApi(input).then((data) => {
                setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: data }]);
            }).finally(() => {
                setLoading(false); // Set loading to false when the API call is complete
            });
        }
    }

    return (
       <SidebarProvider>

<div className="flex h-screen w-full">
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

  <div className="flex-1 p-4 flex flex-col">
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
        </form>
      </div>
    ) : (
      <div className="flex flex-col h-full">
        <h2 className="text-2xl font-bold mb-4">Chat</h2>
        <div className="flex-1 overflow-y-auto mb-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`p-2 rounded-lg ${message.role === 'user' ? 'bg-blue-100 ml-auto' : 'bg-gray-100'
                } max-w-[80%]`}
            >
              {message.content}
            </div>
          ))}
          {loading && (
            <div className="flex justify-center">
              <LoadingSpinner size={32} className="text-gray-500" />
            </div>
          )}
        </div>

        {/* Chat Input Area */}
        <div className="flex items-center w-full mt-auto">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message here..."
            className="flex-1 w-full resize-none"
          />
          <Button
            onClick={handleSendMessage}
            disabled={loading}
            className="ml-2"
          >
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

