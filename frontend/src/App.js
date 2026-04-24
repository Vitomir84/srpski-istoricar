import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import Header from './components/Header';
import ChatContainer from './components/ChatContainer';
import InputContainer from './components/InputContainer';
import DocumentsPanel from './components/DocumentsPanel';
import PeriodSelection from './components/PeriodSelection';
import WelcomeMessage from './components/WelcomeMessage';
import DonationCard from './components/DonationCard';
import SvetiSavaShowcase from './components/SvetiSavaShowcase';

// API URL from environment variable - defaults to proxy for development
const API_URL = process.env.REACT_APP_API_URL || '';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState(null);
  const [periodSelected, setPeriodSelected] = useState(false);
  const [allDocuments, setAllDocuments] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [showWelcome, setShowWelcome] = useState(false);
  const [showPeriodSelection, setShowPeriodSelection] = useState(true);

  const chatContainerRef = useRef(null);

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
    checkServerHealth();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const loadDocuments = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/documents`);
      if (response.data.documents) {
        setAllDocuments(response.data.documents);
      }
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const checkServerHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`);
      console.log('Server status:', response.data);
    } catch (error) {
      console.error('Server not reachable:', error);
    }
  };

  const handlePeriodSelect = (period) => {
    setSelectedPeriod(period === 'svi' ? null : period);
    setPeriodSelected(true);
    setShowWelcome(true);
  };

  const handleSendMessage = async (message) => {
    if (!message.trim() || !periodSelected) return;

    // Add user message
    const userMessage = { content: message, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const requestBody = {
        message: message,
        period: selectedPeriod
      };

      if (selectedDocuments.length > 0) {
        requestBody.selected_documents = selectedDocuments;
      }

      const response = await axios.post(`${API_URL}/api/chat`, requestBody);
      
      // Add agent message
      const agentMessage = { content: response.data.response, isUser: false };
      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      const errorMessage = {
        content: `Грешка при комуникацији са сервером: ${error.message}`,
        isUser: false
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentToggle = (filename, isSelected) => {
    if (isSelected) {
      if (!selectedDocuments.includes(filename)) {
        setSelectedDocuments(prev => [...prev, filename]);
      }
    } else {
      setSelectedDocuments(prev => prev.filter(f => f !== filename));
    }
  };

  const handleSelectAll = () => {
    setSelectedDocuments(allDocuments.map(d => d.filename));
  };

  const handleDeselectAll = () => {
    setSelectedDocuments([]);
  };

  return (
    <div className="app-container">
      <div className="container">
        <div className={`main-chat-area${periodSelected ? ' period-unlocked' : ''}`}>
          <Header 
            selectedPeriod={selectedPeriod} 
            periodSelected={periodSelected}
          />
          <ChatContainer
            ref={chatContainerRef}
            messages={messages}
            isLoading={isLoading}
            showPeriodSelection={showPeriodSelection}
            showWelcome={showWelcome}
            onPeriodSelect={handlePeriodSelect}
          >
            {showPeriodSelection && (
              <PeriodSelection
                onSelect={handlePeriodSelect}
                activePeriod={selectedPeriod}
                periodSelected={periodSelected}
              />
            )}
            {showWelcome && !showPeriodSelection && messages.length === 0 && (
              <WelcomeMessage />
            )}
          </ChatContainer>

          <InputContainer
            onSendMessage={handleSendMessage}
            disabled={!periodSelected}
            isLoading={isLoading}
          />
        </div>

        <DocumentsPanel
          documents={allDocuments}
          selectedDocuments={selectedDocuments}
          onDocumentToggle={handleDocumentToggle}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />
      </div>
      <DonationCard />
    </div>
  );
}

export default App;
