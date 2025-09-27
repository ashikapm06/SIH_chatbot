import axios from 'axios';
import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const ChatWidget: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your AI assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };
// ... all your code above remains the same ...

const handleSendMessage = async () => {
  if (!inputValue.trim()) return;
  const userMsg: Message = {
    id: Date.now().toString(),
    text: inputValue,
    sender: 'user',
    timestamp: new Date(),
  };
  setMessages(prev => [...prev, userMsg]);
  setInputValue('');

  try {
    
    const response = await axios.post('https://fastapi-chatbot-x91y.onrender.com/chat', {
      query: inputValue
    });
    const botText = response.data.answer || "Sorry, I didn't understand that.";
    const botMsg: Message = {
      id: (Date.now() + 1).toString(),
      text: botText,
      sender: 'bot',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, botMsg]);
  } catch (error) {
    const botMsg: Message = {
      id: (Date.now() + 2).toString(),
      text: "Error: Could not connect to server.",
      sender: 'bot',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, botMsg]);
  }
};


  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSendMessage();
  };

  return (
    <div className="fixed inset-0 flex flex-col bg-white text-gray-900 z-50" style={{ minHeight: '100vh' }}>
      {/* Header */}
      <div className="py-5 px-8 border-b flex items-center justify-between">
        <h3 className="font-semibold text-xl">AI Assistant</h3>
      </div>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map(message => (
          <div key={message.id} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className="flex items-start gap-3 max-w-[85%]">
              {message.sender === 'bot' && (
                <div className="w-9 h-9 bg-white-600 rounded-full flex items-center justify-center text-white text-base font-medium"></div>
              )}
              <div className="flex flex-col">
                <div className={`px-5 py-4 rounded-2xl ${message.sender === 'user' ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
                  <p className="text-base">{message.text}</p>
                </div>
                <span className="text-xs text-gray-400 mt-1 px-2">{formatTime(message.timestamp)}</span>
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      {/* Input */}
      <div className="p-6 border-t">
        <div className="flex items-center gap-3">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="w-full px-5 py-4 border rounded-full text-base focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim()}
            className="w-11 h-11 bg-purple-600 text-white rounded-full flex items-center justify-center hover:bg-purple-700 disabled:bg-gray-300 transition duration-200"
          >
            <Send size={21} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWidget;
