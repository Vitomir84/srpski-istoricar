import React, { forwardRef } from 'react';
import './ChatContainer.css';
import Message from './Message';
import LoadingIndicator from './LoadingIndicator';
import svetiSavaImage from '../assets/SvetiSavaMileseva.jpg';

const ChatContainer = forwardRef(({ messages, isLoading, children }, ref) => {
  return (
    <div className="chat-container" ref={ref} style={{
      backgroundImage: `url(${svetiSavaImage})`
    }}>
      {children}
      {messages.map((message, index) => (
        <Message key={index} content={message.content} isUser={message.isUser} />
      ))}
      {isLoading && <LoadingIndicator />}
    </div>
  );
});

export default ChatContainer;
